use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use cubecl::prelude::*;
use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
use smolgrad_rs::core::backend::kernels::binary;
use smolgrad_rs::core::backend::{Backend, Buffer};

type TestRuntime = WgpuRuntime;

fn get_test_client() -> ComputeClient<TestRuntime> {
    let device = WgpuDevice::default();
    WgpuRuntime::client(&device)
}

fn bench_binary_ops(c: &mut Criterion) {
    let client = get_test_client();
    let backend = Backend::new(client.clone());

    let sizes = [1024, 1024 * 1024];
    let mut group = c.benchmark_group("binary_ops");

    for size in sizes {
        let shape = vec![size];
        let lhs = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, shape.clone());
        let rhs = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, shape.clone());
        let mut out = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, shape.clone());

        let element_size = std::mem::size_of::<f32>() as u64;
        group.throughput(Throughput::Bytes(size as u64 * element_size * 3)); // 2 reads, 1 write

        group.bench_function(format!("add_size_{}", size), |b| {
            b.iter(|| {
                binary::launch_add(&client, &lhs, &rhs, &mut out, 1);
            });
        });

        group.bench_function(format!("add_size_{}_vec4", size), |b| {
            b.iter(|| {
                binary::launch_add(&client, &lhs, &rhs, &mut out, 4);
            });
        });

        group.bench_function(format!("sub_size_{}", size), |b| {
            b.iter(|| {
                binary::launch_sub(&client, &lhs, &rhs, &mut out, 1);
            });
        });

        group.bench_function(format!("mul_size_{}", size), |b| {
            b.iter(|| {
                binary::launch_mul(&client, &lhs, &rhs, &mut out, 1);
            });
        });

        group.bench_function(format!("div_size_{}", size), |b| {
            b.iter(|| {
                binary::launch_div(&client, &lhs, &rhs, &mut out, 1);
            });
        });
    }
    group.finish();
}

fn bench_broadcasting(c: &mut Criterion) {
    let client = get_test_client();
    let backend = Backend::new(client.clone());

    let mut group = c.benchmark_group("broadcasting");

    let n = 1024;
    // [1024, 1] + [1, 1024] -> [1024, 1024]
    let lhs = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, vec![n, 1]);
    let rhs = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, vec![1, n]);
    let mut out = Buffer::<TestRuntime, f32>::zeros_with_backend(&backend, vec![n, n]);

    let element_size = std::mem::size_of::<f32>() as u64;
    group.throughput(Throughput::Bytes((n * n) as u64 * element_size * 3));

    group.bench_function("add_broadcast_1024x1024", |b| {
        b.iter(|| {
            binary::launch_add(&client, &lhs, &rhs, &mut out, 1);
        });
    });

    group.finish();
}

criterion_group!(benches, bench_binary_ops, bench_broadcasting);
criterion_main!(benches);
