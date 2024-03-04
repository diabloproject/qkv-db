use std::ops::Add;
use std::path::PathBuf;
use ndarray::{Array2};
use rand::{Rng, thread_rng};
extern crate blas_src;
mod storage;



fn softmax(array: &Array2<f32>) -> Array2<f32> {
    let exp_array = array.mapv(|x| x.exp());
    let sum = exp_array.sum_axis(Axis(0));
    exp_array / sum
}

#[tokio::main]
pub async fn main() -> anyhow::Result<()> {
    let config = storage::DatabaseConfiguration {
        qkv_vec_size: 512,
    };
    let bucket_path = PathBuf::from("./data/b/a");
    let mut bucket = storage::Bucket::from_disk(&bucket_path, &config).await?;
    bucket.clear().await?;
    println!("Inserting fake data");
    for i in 0..1024 {
        println!("{i} of 1024...");
        bucket.insert_kv((0..4096).map(|_| (
            thread_rng().sample_iter(rand::distributions::Standard).take(512).collect(),
            thread_rng().sample_iter(rand::distributions::Standard).take(512).collect()
        )).collect()).await?;
    }
    let q: Array2<f32> = Array2::from_shape_vec((2, 512), thread_rng().sample_iter(rand::distributions::Standard).take(1024).collect::<Vec<f32>>()).expect("Unable to initialize array");
    let mut res_batched = Array2::from_shape_vec((2, 512), vec![0.; 1024]).unwrap();
    let batch_size = num_cpus::get() * 1024;
    let time = std::time::Instant::now();
    // Size of the matrices are ~4gb
    // println!("Starting processing");
    // bucket.reduce_kv_batched(&mut res_batched, batch_size, compute_cross_attention(512, &q)).await;
    // println!("Scan finished in {:?}", time.elapsed()); // 3 secs
    Ok(())
}

use ndarray::prelude::*;



