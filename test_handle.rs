use cubecl::server::Handle;
fn main() {
    let mut handle: Handle = unsafe { std::mem::zeroed() };
    handle.offset = 0;
}
