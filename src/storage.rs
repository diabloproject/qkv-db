use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter, write};
use std::io::SeekFrom;
use std::marker::PhantomData;
use std::mem::size_of;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use tokio::fs::File;
use tokio::io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt, BufReader};

#[derive(Debug, Copy, Clone)]
pub struct InvalidLayoutError;
impl Display for InvalidLayoutError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Error for InvalidLayoutError {}


pub struct VecView<'parent, T: Sized> {
    v: *const T,
    size: usize,
    phd: PhantomData<&'parent Vec<u8>>
}

impl<'p, T: Sized> VecView<'p, T> {
    pub fn from_vec(vec: &'p Vec<u8>) -> Result<Self, InvalidLayoutError>{
        if vec.len() % size_of::<T>() != 0 {
            return Err(InvalidLayoutError)
        }
        Ok(Self {
            v: vec.as_ptr() as *const T,
            size: vec.len() / size_of::<T>(),
            phd: PhantomData::default(),
        })
    }

    pub fn get(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.v, self.size) }
    }
}

impl<'p, T> Deref for VecView<'p, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.v, self.size) }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct DatabaseConfiguration {
    pub qkv_vec_size: u32,
}

pub struct Row {}

pub struct Bucket {
    keys_handle: File,
    values_handle: File,
    qkv_vec_size: u32,
}

impl Bucket {
    pub async fn initialize(path: &Path, database_config: DatabaseConfiguration) -> Result<Bucket, std::io::Error> {
        tokio::fs::create_dir_all(path).await?;
        Ok(Self {
            keys_handle: File::options().write(true).read(true).create(true).open(path.join("keys.bin")).await?,
            values_handle: File::options().write(true).read(true).create(true).open(path.join("values.bin")).await?,
            qkv_vec_size: database_config.qkv_vec_size,
        })
    }

    pub async fn from_disk(path: &Path, database_config: &DatabaseConfiguration) -> Result<Bucket, std::io::Error> {
        Ok(Self {
            keys_handle: File::options().write(true).read(true).open(path.join("keys.bin")).await?,
            values_handle: File::options().write(true).read(true).open(path.join("values.bin")).await?,
            qkv_vec_size: database_config.qkv_vec_size,
        })
    }
    pub async fn reduce_kv_batched<A: ?Sized, F: Fn(&mut A, &[f32], &[f32]) -> ()>(&mut self, acc: &mut A, batch_size: usize, f: F) {
        self.keys_handle.seek(SeekFrom::Start(0)).await.expect("I/O error occurred during bucket keys read.");
        self.values_handle.seek(SeekFrom::Start(0)).await.expect("I/O error occurred during bucket values read.");

        // Define buffers and load first block into memory
        let mut keys_buf: Vec<u8> = Vec::with_capacity(size_of::<f32>() * self.qkv_vec_size as usize * batch_size);
        let mut values_buf: Vec<u8> = Vec::with_capacity(size_of::<f32>() * self.qkv_vec_size as usize * batch_size);

        // read_buf does NOT extend capacity, so after these reads buffers contain less or equal to self.qkv_vec_size * READ_BLOCK_SIZE floats.
        self.keys_handle.read_buf(&mut keys_buf).await.expect("I/O error occurred during bucket values read.");
        self.values_handle.read_buf(&mut values_buf).await.expect("I/O error occurred during bucket values read.");
        loop {
            // Allows us to obtain &[f32] from Vec<u8> without allocations
            let keys: VecView<f32> = VecView::from_vec(&keys_buf).unwrap();
            let values: VecView<f32> = VecView::from_vec(&values_buf).unwrap();
            if keys.len() == 0 {
                // Data file is ended.
                break;
            }

            f(acc, keys.as_ref(), values.as_ref());

            // Required because read_buf adds to existing buffer instead of rewriting from scratch.
            keys_buf.clear();
            values_buf.clear();

            self.keys_handle.read_buf(&mut keys_buf).await.expect("I/O error occurred during bucket values read.");
            self.values_handle.read_buf(&mut values_buf).await.expect("I/O error occurred during bucket values read.");
        }
    }

    pub async fn insert_kv(&mut self, data: Vec<(Vec<f32>, Vec<f32>)>) -> Result<(), std::io::Error> {
        let mut keys_to_be_written = Vec::with_capacity(data.len() * self.qkv_vec_size as usize);
        let mut values_to_be_written = Vec::with_capacity(data.len() * self.qkv_vec_size as usize);
        for (mut k, mut v) in data.into_iter() {
            keys_to_be_written.append(&mut k);
            values_to_be_written.append(&mut v);
        }
        self.keys_handle.seek(SeekFrom::End(0)).await.expect("I/O error occurred during bucket values read.");
        self.values_handle.seek(SeekFrom::End(0)).await.expect("I/O error occurred during bucket values read.");
        let keys_bytes = unsafe { std::slice::from_raw_parts(keys_to_be_written.as_ptr() as *const u8, keys_to_be_written.len() * size_of::<f32>()) };
        let values_bytes = unsafe { std::slice::from_raw_parts(values_to_be_written.as_ptr() as *const u8, values_to_be_written.len() * size_of::<f32>()) };
        self.keys_handle.write(keys_bytes).await?;
        self.values_handle.write(values_bytes).await?;
        self.keys_handle.flush().await?;
        self.values_handle.flush().await?;
        Ok(())
    }

    pub async fn clear(&mut self) -> Result<(), std::io::Error>{
        self.keys_handle.set_len(0).await?;
        self.values_handle.set_len(0).await?;
        self.keys_handle.seek(SeekFrom::Start(0)).await?;
        self.values_handle.seek(SeekFrom::Start(0)).await?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct AlreadyInUse { name: String, ty: String }
impl Display for AlreadyInUse {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} already in use.", self.ty, self.name)
    }
}

impl Error for AlreadyInUse {}


#[derive(Debug, Clone)]
pub struct AlreadyExists { name: String, ty: String }
impl Display for AlreadyExists {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} already in exists.", self.ty, self.name)
    }
}

impl Error for AlreadyExists {}

pub struct Database {
    data_directory: PathBuf,
    buckets: HashMap<Arc<str>, Bucket>,
    conf: DatabaseConfiguration,
}

impl Database {
    pub fn get_qkv_vec_size(&self) -> u32 {
        self.conf.qkv_vec_size
    }
}

impl Database {
    pub async fn from_disk(data_directory: PathBuf) -> Result<Database, std::io::Error> {
        let content = tokio::fs::read_to_string(data_directory.join("bucket_info.index")).await?;
        let buf = tokio::fs::read(data_directory.join("conf.bc")).await?;
        let conf = bincode::deserialize(&buf).expect(&format!("Configuration file of database {} is corrupted. Unable to initialize database.", data_directory.file_name().unwrap().to_str().unwrap()));
        let bucket_names: Vec<&str> = content.split("\n").filter(|x| !x.is_empty()).collect();
        let mut buckets: HashMap<Arc<str>, Bucket> = Default::default();
        for name in bucket_names {
            buckets.insert(Arc::from(name), Bucket::from_disk(&data_directory.join(&name), &conf).await?);
        }
        Ok(Self {
            data_directory, buckets, conf
        })
    }

    pub async fn get_bucket(&mut self, name: &str) -> Result<Option<&mut Bucket>, AlreadyInUse> {
        match self.buckets.get_mut(name) {
            None => {Ok(None)}
            Some(b) => {
                Ok(Some(b))
            }
        }
    }

    pub async fn create_bucket(&mut self, name: &str) -> Result<(), AlreadyExists> {
        if self.buckets.keys().find(|x| x.as_ref() == name).is_some() {
            return Err(AlreadyExists {
                name: "Bucket".to_string(),
                ty: name.to_string(),
            });
        }
        self.buckets.insert(name.into(), Bucket::initialize(&self.data_directory.join(name), self.conf).await.unwrap());
        tokio::fs::write(self.data_directory.join("bucket_info.index"), self.buckets.keys().map(|k| k.to_string()).collect::<Vec<String>>().join("\n")).await.unwrap();
        Ok(())
    }

    pub async fn initialize(data_directory: &Path, database_configuration: DatabaseConfiguration) -> Result<Database, std::io::Error> {
        tokio::fs::create_dir_all(data_directory).await?;
        tokio::fs::write(data_directory.join("bucket_info.index"), []).await?;
        tokio::fs::write(data_directory.join("conf.bc"), bincode::serialize(&database_configuration).unwrap()).await?;
        Ok(Self {
            data_directory: data_directory.into(),
            buckets: Default::default(),
            conf: database_configuration,
        })
    }
}

pub struct Storage {
    data_directory: PathBuf,
    databases: HashMap<Arc<str>, Database>
}

impl Storage {
    pub async fn from_disk(data_directory: PathBuf) -> Result<Storage, std::io::Error> {
        let content = tokio::fs::read_to_string(data_directory.join("db_info.index")).await?;
        let database_names: Vec<&str> = content.split("\n").filter(|x| !x.is_empty()).collect();
        let mut databases: HashMap<Arc<str>, Database> = Default::default();
        for name in database_names {
            databases.insert(name.into(), Database::from_disk(data_directory.join(name)).await?);
        };
        Ok(Self {
            data_directory,
            databases
        })
    }

    pub async fn create_database(&mut self, name: &str, database_configuration: DatabaseConfiguration) -> Result<(), AlreadyExists> {
        if self.databases.keys().find(|x| x.as_ref() == name).is_some() {
            return Err(AlreadyExists {
                name: "Database".to_string(),
                ty: name.to_string(),
            });
        };
        self.databases.insert(name.into(), Database::initialize(&self.data_directory.join(name), database_configuration).await.unwrap());
        tokio::fs::write(self.data_directory.join("db_info.index"), self.databases.keys().map(|k| k.to_string()).collect::<Vec<String>>().join("\n")).await.unwrap();
        Ok(())
    }
    pub async fn get_database(&mut self, name: &str) -> Result<Option<&mut Database>, AlreadyInUse> {
        match self.databases.get_mut(name) {
            None => {Ok(None)}
            Some(b) => {
                Ok(Some(b))
            }
        }
    }
}