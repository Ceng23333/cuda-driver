mod blob;
mod gguf;
mod loader;

use blob::{Blob, Data};
use cuda::Device;
use gguf::{GGufModel, map_files};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use loader::{MemCalculator, WeightLoader};
use nn::{Dim, Edge, External, GraphBuilder, Session, TensorMeta, op};
use std::{
    collections::{BTreeMap, HashMap},
    fmt,
    ops::Range,
};
use tensor::Tensor;

// cargo run --release -- ../TinyStory-5M-v0.0-F32.gguf
fn main() {
    if cuda::init().is_err() {
        return;
    }

    let path = std::env::args_os().nth(1).unwrap();
    let maps = map_files(path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    insert_sin_cos(&mut gguf);

    let nvoc = 32000usize;
    let nctx = meta![gguf => llm_context_length];
    let nblk = meta![gguf => llm_block_count];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let nkvh = meta![gguf => llm_attention_head_count_kv; nh];
    let dh = meta![gguf => llm_rope_dimension_count; d / nh];
    let di = meta![gguf => llm_feed_forward_length];
    let epsilon = meta![gguf => llm_attention_layer_norm_rms_epsilon; 1e-5];

    let llama = ::nn::LLaMA {
        embedding: ::nn::Embedding {
            dt: types::F32,
            d: d.into(),
            wte: ::nn::Table {
                row: nvoc.into(),
                weight: "token_embd.weight".to_string(),
            },
            wpe: None,
        },
        blks: (0..nblk)
            .map(|iblk| ::nn::TransformerBlk {
                attn_norm: ::nn::Normalization {
                    d: d.into(),
                    epsilon: epsilon as _,
                    items: ::nn::NormType::RmsNorm {
                        dt: types::F32,
                        scale: format!("blk.{iblk}.attn_norm.weight"),
                    },
                },
                attn: ::nn::Attention {
                    nh: nh.into(),
                    nkvh: nkvh.into(),
                    qkv: ::nn::Linear {
                        dt: types::F32,
                        shape: [((nh + nkvh + nkvh) * dh).into(), d.into()],
                        weight: format!("blk.{iblk}.attn_qkv.weight"),
                        bias: None,
                    },
                    rope: Some(::nn::RoPE {
                        nctx: nctx.into(),
                        sin: "sin_table".into(),
                        cos: "cos_table".into(),
                    }),
                    sessions: [Session {
                        seq: Dim::var("n"),
                        cache: None,
                    }]
                    .into(),
                    output: ::nn::Linear {
                        dt: types::F32,
                        shape: [d.into(), (nh * dh).into()],
                        weight: format!("blk.{iblk}.attn_output.weight"),
                        bias: None,
                    },
                },
                ffn_norm: ::nn::Normalization {
                    d: d.into(),
                    epsilon: epsilon as _,
                    items: ::nn::NormType::RmsNorm {
                        dt: types::F32,
                        scale: format!("blk.{iblk}.ffn_norm.weight"),
                    },
                },
                ffn: ::nn::Mlp {
                    up: ::nn::Linear {
                        dt: types::F32,
                        shape: [(di * 2).into(), d.into()],
                        weight: format!("blk.{iblk}.ffn_gate_up.weight"),
                        bias: None,
                    },
                    act: ::nn::Activation::SwiGLU,
                    down: ::nn::Linear {
                        dt: types::F32,
                        shape: [d.into(), di.into()],
                        weight: format!("blk.{iblk}.ffn_down.weight"),
                        bias: None,
                    },
                },
            })
            .collect(),
    };

    let ::nn::Graph { topo, nodes, edges } = GraphBuilder::default()
        .register_op("embedding", op::embedding::Embedding)
        .register_op("rms-norm", op::normalization::RmsNorm)
        .register_op("layer-norm", op::normalization::LayerNorm)
        .register_op("attention", op::attention::Attention)
        .register_op("split", op::split::Split)
        .register_op("swiglu", op::activation::SwiGLU)
        .register_op("gelu", op::activation::GeLU)
        .register_op("linear", op::linear::Linear)
        .register_op("rope", op::rope::Rope)
        .register_op("concat", op::concat::Concat)
        .build(
            llama,
            [
                TensorMeta::new(types::U32, [Dim::var("n")]),
                TensorMeta::new(types::U32, [Dim::var("n")]),
            ],
        )
        .unwrap();

    for (i, topo) in topo.iter().enumerate() {
        println!(
            "{i:>3}. {:10} {:40} {:?} <- {:?}",
            nodes[i].op, nodes[i].name, topo.outputs, topo.inputs
        )
    }

    const N: usize = 5;
    const ALIGN: usize = 512;

    let edges = edges
        .into_iter()
        .map(|edge| fix_n(edge, &gguf, N))
        .collect::<Box<_>>();

    let mut weights = HashMap::<*const u8, Range<usize>>::new();
    let mut calculator = MemCalculator::new(ALIGN);
    let mut sizes = BTreeMap::<usize, usize>::new();
    for edge in &edges {
        if let TensorData::Weight(_, data) = edge.get() {
            let ptr = data.as_ptr();
            let len = data.len();

            use std::collections::hash_map::Entry::{Occupied, Vacant};
            match weights.entry(ptr) {
                Occupied(entry) => {
                    assert_eq!(entry.get().len(), len)
                }
                Vacant(entry) => {
                    entry.insert(calculator.push(len));
                    *sizes.entry(len).or_insert(0) += 1
                }
            }
        }
    }

    Device::new(0).context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            sizes
                .iter()
                .filter(|(_, times)| **times < nblk)
                .map(|(&size, _)| size),
        );
        let mut weight_memory = ctx.malloc::<u8>(calculator.size());

        let stream = ctx.stream();
        for edge in &edges {
            if let TensorData::Weight(_, data) = edge.get() {
                let range = weights[&data.as_ptr()].clone();
                let dev = &mut weight_memory[range.clone()];
                loader.load(dev, data, &stream);
                // Tensor::<usize, 2>::from_dim_slice(edge.dt, edge.shape).map(|_| range)
            }
        }
        //     let weights = llama
        //         .map(|t| {

        //         })
        //         .map(|t| t.map(|range| &weight_memory[range]));

        //     let graph = Graph::new();
        //     let mut modules = Modules::new(ctx);
        //     weights.add_to_graph(&graph, &mut modules, &workspace);

        //     let mut workspace = workspace.map(|vir| vir.map_on(&dev));
        //     let tokens = workspace.tokens.get().clone();
        //     memcpy_h2d(
        //         &mut workspace.item[tokens],
        //         &[4612u32, 2619, 260, 647, 31844],
        //     );

        //     ctx.instantiate(&graph).launch(&stream);
        //     stream.synchronize();

        //     let x = workspace.x.clone().map(|range| {
        //         let mut host = vec![0u8; range.len()];
        //         memcpy_d2h(&mut host, &workspace.item[range]);
        //         host
        //     });
        //     println!("{}", Fmt(x.as_deref()))
    })
}

/// 构造 sin cos 表张量，存储到 GGufModel 中
fn insert_sin_cos(gguf: &mut GGufModel) {
    let nctx = meta![gguf => llm_context_length];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let dh = meta![gguf => llm_rope_dimension_count; d / nh];
    let theta = meta![gguf => llm_rope_freq_base; 1e4];

    let ty = types::F32;
    let mut sin = Blob::new(nctx * dh / 2 * ty.nbytes());
    let mut cos = Blob::new(nctx * dh / 2 * ty.nbytes());

    {
        let ([], sin, []) = (unsafe { sin.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        let ([], cos, []) = (unsafe { cos.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        for pos in 0..nctx {
            for i in 0..dh / 2 {
                let theta = theta.powf(-((2 * i) as f32 / dh as f32));
                let freq = pos as f32 * theta;
                let (sin_, cos_) = freq.sin_cos();
                sin[pos * dh / 2 + i] = sin_;
                cos[pos * dh / 2 + i] = cos_;
            }
        }
    }

    let mut insert = |name, data: Blob| {
        assert!(
            gguf.tensors
                .insert(
                    name,
                    Tensor::from_dim_slice(ty, [nctx, dh / 2]).map(|len| {
                        assert_eq!(len, data.len());
                        data.into()
                    })
                )
                .is_none()
        )
    };
    insert("sin_table", sin);
    insert("cos_table", cos);
}

pub enum TensorData<'a> {
    Weight(String, &'a Data<'a>),
    Normal(usize),
}

fn fix_n<'a>(edge: Edge<String>, gguf: &'a GGufModel, n: usize) -> Tensor<TensorData<'a>, 3> {
    let value = HashMap::from([("n", n)]);

    let Edge {
        meta,
        external: weight_info,
    } = edge;
    let TensorMeta { dt, shape } = meta;
    let shape = shape
        .iter()
        .map(|d| d.substitute(&value))
        .collect::<Vec<_>>();

    match weight_info {
        Some(External { name, item }) => {
            let tensor = &gguf.tensors[&*item];
            assert_eq!(tensor.dt(), dt, "dt mismatch: {name}");
            assert_eq!(tensor.shape(), &shape, "shape mismatch: {name}");
            tensor.as_ref().map(|data| TensorData::Weight(name, data))
        }
        None => Tensor::from_dim_slice(dt, &*shape).map(TensorData::Normal),
    }
}

struct Fmt<'a, const N: usize>(Tensor<&'a [u8], N>);

impl<const N: usize> fmt::Display for Fmt<'_, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = self.0.layout();
        let ptr = self.0.get().as_ptr();
        match self.0.dt() {
            types::F32 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<f32>>()) },
            types::U32 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<u32>>()) },
            _ => todo!(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DataFmt<T>(T);

impl fmt::Display for DataFmt<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}
