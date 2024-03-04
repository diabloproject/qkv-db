mod command;
mod storage;

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Display, Formatter, write};
use std::io::{Error, SeekFrom, Write};
use std::ops::{Add, Not};
use std::path::PathBuf;
use std::sync::Arc;
use ndarray::{Array2, Axis};
use thiserror::Error;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt};
use tokio::sync::RwLock;
use crate::command::{Command, ParseError, PropertyValue, ScanTargetBucket};
use crate::storage::{AlreadyInUse, DatabaseConfiguration, Storage};
use ndarray::prelude::*;
use tokio::net::TcpListener;

extern crate blas_src;

fn compute_cross_attention<'a>(qkv_vec_size: usize, q: &'a Array2<f32>) -> impl Fn(&mut Array2<f32>, &[f32], &[f32]) + 'a {
    return move |mut acc: &mut Array2<f32>, k: &[f32], v: &[f32]| {
        let k_vec: Vec<f32> = Vec::from(k);
        let v_vec: Vec<f32> = Vec::from(v);
        let k: Array2<f32> = Array2::from_shape_vec((k_vec.len() / qkv_vec_size, qkv_vec_size), k_vec).unwrap();  // 1x4
        let v: Array2<f32> = Array2::from_shape_vec((v_vec.len() / qkv_vec_size, qkv_vec_size), v_vec).unwrap();  // 1x4
        let e_scores = q.dot(&k.t()).mapv(|x| x.exp());
        let scores = e_scores.clone() / e_scores.sum_axis(Axis(0));
        let re_ = scores.dot(&v);
        *acc = re_.add(acc.clone() / qkv_vec_size as f32);
    };
}

#[derive(Debug, Copy, Clone)]
pub enum EntityType {
    Database,
    Bucket,
}

impl Display for EntityType {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            EntityType::Database => { write!(f, "database") }
            EntityType::Bucket => { write!(f, "bucket") }
        }
    }
}

#[derive(Debug, Clone, Error)]
pub enum ExecutionError {
    #[error("Database '{database}' does not exist")]
    DatabaseDoesNotExist { database: String },
    #[error("Bucket '{bucket}' does not exist inside database '{database}'")]
    BucketDoesNotExist { database: String, bucket: String },
    #[error("Size of received vector does not match the configured database's : ")]
    SizeMismatch { expected: u32, got: u32 },
    #[error("Entity {name} of type {ty} already exists")]
    EntityAlreadyExists {
        name: String,
        ty: EntityType,
    },
    #[error("Property {property} has type {expected}, but {found} was passed")]
    TypeMismatch {
        expected: &'static str,
        found: &'static str,
        property: &'static str,
    },
}

#[derive(Parser, Debug)]
pub struct Args {
    #[arg(
    long,
    default_value = "./qkv-config.json",
    long_help = "Path to database configuration file."
    )]
    pub config: PathBuf,
    #[arg(long, default_value = None, long_help = "Path to command list that will be executed during initialization.")]
    pub init: Option<PathBuf>,
}

#[derive(Serialize, Deserialize, Debug, Default)]
pub struct Configuration {
    data_directory: PathBuf,
}

pub enum Bucket<'a> {
    Hot,
    Name(&'a str),
    All,
}

pub struct Database {
    /// Size of the individual q, k or v vector in the database.
    vec_size: u32,
    /// Buckets that exist in this database besides two virtual ones (`hot` and `all`)
    buckets: Vec<String>,
    /// Directory containing database files
    data_directory: PathBuf,
}

impl Database {
    /// Scan the bucket `b`.
    /// Takes in query and returns attended result across bucket.
    pub fn scan(&self, q: Vec<f32>, b: Bucket<'_>) -> Vec<f32> {
        todo!()
    }

    /// Store the entry in the bucket `bucket_name`
    pub fn store(&mut self, k: Vec<f32>, v: Vec<f32>, bucket_name: &'_ str) {
        todo!()
    }

    pub async fn load_from_path(path: impl Into<PathBuf>) -> Result<Self, InitializationError> {
        let path = path.into();
        let info_file = path.join("bc_info.index");
        let content = tokio::fs::read_to_string(info_file).await?;
        let mut buckets = vec![];
        for bucket in content.split('\n') {
            if bucket == "" {
                continue;
            };
            let bucket_path = path.join(format!("{}.kvs", &bucket));
            if bucket_path.exists().not() || bucket_path.is_file().not() {
                return Err(InitializationError::new_inconsistency_error(
                    bucket_path.display().to_string(),
                    "file",
                    if bucket_path.exists() {
                        "directory"
                    } else {
                        "nto exists"
                    },
                ));
            };
            buckets.push(bucket.to_string())
        }
        Ok(Self {
            buckets,
            data_directory: path,
            vec_size: 32,
        })
    }

    pub async fn init_database(
        path: impl Into<PathBuf>,
        vec_size: u32,
    ) -> Result<Self, InitializationError> {
        let path = path.into();
        tokio::fs::create_dir_all(&path).await?;
        tokio::fs::write(path.join("bc_info.index"), "").await?;
        Ok(Self {
            vec_size,
            buckets: vec![],
            data_directory: path,
        })
    }
}

#[derive(Error, Debug)]
pub enum InitializationError {
    #[error("Expected `{0}` to exist and be a `{1}`, but instead received `{2}`. That means that database is corrupted. You can try to fix yourself, but you probably should restore the backup.")]
    InconsistentDataDirectory(String, String, String),
    #[error("{0}")]
    IOError(std::io::Error),
}

impl From<std::io::Error> for InitializationError {
    fn from(value: Error) -> Self {
        return Self::IOError(value);
    }
}

impl InitializationError {
    pub fn new_inconsistency_error(
        inconsistent_object_name: impl Into<String>,
        required_state: impl Into<String>,
        found_state: impl Into<String>,
    ) -> Self {
        InitializationError::InconsistentDataDirectory(
            inconsistent_object_name.into(),
            required_state.into(),
            found_state.into(),
        )
    }
}

pub struct Engine {
    storage: Storage,
}

impl Engine {
    pub async fn new(conf: Configuration) -> Self {
        Self {
            storage: Storage::from_disk(conf.data_directory).await.unwrap()
        }
    }

    pub async fn create_database(
        &mut self,
        name: String,
        vec_size: u32,
    ) -> Result<(), ExecutionError> {
        self.storage.create_database(&name, DatabaseConfiguration {
            qkv_vec_size: vec_size
        }).await.unwrap();
        Ok(())
    }

    pub async fn execute(&mut self, command: Command) -> Result<Option<Vec<Vec<f32>>>, ExecutionError> {
        match command {
            Command::CreateDatabase { name, properties } => {
                if self.storage.get_database(&name).await.unwrap().is_some() {
                    return Err(ExecutionError::EntityAlreadyExists { name, ty: EntityType::Database });
                }

                let prop = properties.iter().find(|x| x.name == "qkv_vec_size").map(|x| x.clone().data);
                let qkv_vec_size = match prop {
                    Some(v) => match v {
                        PropertyValue::Integer(v) => Ok(v),
                        PropertyValue::Float(_) => Err(ExecutionError::TypeMismatch { expected: "Unsigned integer", found: "Float", property: "qkv_vec_size" }),
                        PropertyValue::String(_) => Err(ExecutionError::TypeMismatch { expected: "Unsigned integer", found: "String", property: "qkv_vec_size" })
                    }
                    None => Ok(512)
                }?;
                if qkv_vec_size <= 0 {
                    return Err(ExecutionError::TypeMismatch {
                        expected: "Unsigned integer",
                        found: "Signed integer",
                        property: "qkv_vec_size",
                    });
                }
                self.create_database(name, qkv_vec_size as u32).await?;
                Ok(None)
            }
            Command::CreateBucket { database, name, properties } => {
                self.create_bucket(&name, &database).await?;
                Ok(None)
            }
            Command::Insert { database, bucket, entries, properties } => {
                // Checking that all vectors have same and valid size
                let target_size = match self.storage.get_database(&database).await.unwrap() {
                    None => { return Err(ExecutionError::DatabaseDoesNotExist { database: database.into() }); }
                    Some(x) => { x.get_qkv_vec_size() }
                };
                for (k, v) in entries.iter() {
                    if target_size != k.len() as u32 {
                        return Err(ExecutionError::SizeMismatch {
                            expected: target_size,
                            got: k.len() as u32,
                        });
                    }

                    if target_size != v.len() as u32 {
                        return Err(ExecutionError::SizeMismatch {
                            expected: target_size,
                            got: k.len() as u32,
                        });
                    }
                }

                self.insert(entries, &bucket, &database).await;
                Ok(None)
            }
            Command::Scan { database, bucket, queries, properties } => {
                let bucket = match bucket {
                    ScanTargetBucket::Hot => { todo!() }
                    ScanTargetBucket::All => { todo!() }
                    ScanTargetBucket::Physical(name) => { name }
                };

                let target_size = match self.storage.get_database(&database).await.unwrap() {
                    None => { return Err(ExecutionError::DatabaseDoesNotExist { database }); }
                    Some(c) => { c }
                }.get_qkv_vec_size();
                for q in queries.iter() {
                    if target_size != q.len() as u32 {
                        return Err(ExecutionError::SizeMismatch { expected: target_size, got: q.len() as u32 });
                    }
                }
                Ok(Some(self.scan(queries, &bucket, &database).await?))
            }
            Command::Dummy => {
                Ok(None)
            }
        }
    }
    async fn create_bucket(&mut self, bucket_name: &str, database: &str) -> Result<(), ExecutionError> {
        match self.storage.get_database(database).await.unwrap() {
            None => { Err(ExecutionError::DatabaseDoesNotExist { database: database.into() }) }
            Some(db) => {
                db.create_bucket(bucket_name).await.unwrap();
                Ok(())
            }
        }
    }
    async fn scan(&mut self, queries: Vec<Vec<f32>>, bucket: &str, database: &str) -> Result<Vec<Vec<f32>>, ExecutionError> {
        match self.storage.get_database(database).await.unwrap() {
            None => { Err(ExecutionError::DatabaseDoesNotExist { database: database.into() }) }
            Some(db) => {
                match db.get_bucket(bucket).await.unwrap() {
                    None => { Err(ExecutionError::BucketDoesNotExist { database: database.into(), bucket: bucket.into() }) }
                    Some(bucket) => {
                        if queries.len() == 0 {
                            return Ok(vec![]);
                        }
                        let q_shape = (queries.len(), queries[0].len());
                        let q_vec: Vec<f32> = queries.into_iter().flatten().collect();
                        let q = Array2::from_shape_vec(q_shape, q_vec).unwrap();
                        let mut res: Array2<f32> = Array2::from_elem(q_shape, 0.);
                        let batch_size = num_cpus::get() * 1024;
                        bucket.reduce_kv_batched(&mut res, batch_size, compute_cross_attention(512, &q)).await;
                        Ok(res.rows().into_iter().map(|r| r.to_vec()).collect())
                    }
                }
            }
        }
    }

    async fn insert(&mut self, data: Vec<(Vec<f32>, Vec<f32>)>, bucket: &str, database: &str) {}
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    if !args.config.exists() {
        tokio::fs::write(
            &args.config,
            serde_json::to_string_pretty(&Configuration {
                data_directory: PathBuf::from("./data"),
            })?,
        )
            .await?;
    }

    let conf: Configuration = serde_json::from_str(
        &tokio::fs::read_to_string(&args.config)
            .await
            .expect("Unable to read configuration file"),
    )
        .expect("Invalid configuration file");

    let mut engine = Engine::new(conf).await;

    if let Some(init_path) = args.init {
        let content = tokio::fs::read_to_string(init_path).await?;
        let commands = command::parse_commands(&content)?;
        for com in commands {
            engine.execute(com).await?;
        }
    }


    let listener = TcpListener::bind("127.0.0.1:7878").await.unwrap();
    loop {
        let (mut stream, address) = listener.accept().await?;
        let mut content_size = [0u8; 4];
        match stream.read_exact(&mut content_size).await {
            Ok(c) => c,
            Err(err) => {
                println!("Connection to {} lost.", address);
                continue;
            }
        };
        let content_size = u32::from_le_bytes(content_size);
        let mut content = Vec::from_iter((0..content_size).map(|_| 0u8));
        match stream.read_exact(&mut content).await {
            Ok(c) => c,
            Err(err) => {
                println!("Connection to {} lost.", address);
                continue;
            }
        };
        let commands_text = String::from_utf8(content)?;
        println!("{commands_text}");
        let commands = match command::parse_commands(&commands_text) {
            Ok(c) => { c }
            Err(err) => {
                let content = err.to_string();
                match stream.write(&(content.as_bytes().len() as u32).to_le_bytes()).await {
                    Ok(c) => { c }
                    Err(_) => {
                        println!("Connection to {} lost.", address);
                        continue;
                    }
                };
                match stream.write(content.as_bytes()).await {
                    Ok(c) => (),
                    Err(_) => {
                        println!("Connection to {} lost.", address);
                        continue;
                    }
                }
                continue;
            }
        };
        let mut res: Option<Vec<Vec<f32>>> = None;
        let mut error = false;
        for command in commands {
            res = match engine.execute(command).await {
                Ok(c) => { c }
                Err(err) => {
                    let content = err.to_string();
                    match stream.write(&(content.as_bytes().len() as u32).to_le_bytes()).await {
                        Ok(c) => { c }
                        Err(_) => {
                            println!("Connection to {} lost.", address);
                            continue;
                        }
                    };
                    match stream.write(content.as_bytes()).await {
                        Ok(c) => (),
                        Err(_) => {
                            println!("Connection to {} lost.", address);
                            continue;
                        }
                    }
                    error = true;
                    continue;
                }
            };
        }

        if error {
            continue
        }

        let mut result = String::new();

        if let Some(res) = res {
            result.extend(format!("({})\n", res.into_iter().map(|v| format!("[{}]", v.into_iter().map(|r| r.to_string()).collect::<Vec<String>>().join(", "))).collect::<Vec<String>>().join(", ")).chars());
        }
        result.extend("DONE.".chars());
        match stream.write(&(result.len() as u32).to_le_bytes()).await {
            Ok(c) => (),
            Err(_) => {
                println!("Connection to {address} lost");
                continue;
            }
        };

        match stream.write(result.as_bytes()).await {
            Ok(c) => (),
            Err(_) => {
                println!("Connection to {address} lost.");
                continue;
            }
        };

        stream.flush().await?;
    }

    Ok(())
}
