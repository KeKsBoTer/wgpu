
@group(0) @binding(0)
var<uniform> transform: mat4x4<f32>;

@group(0) @binding(1)
var r_color: texture_2d<u32>;

@group(0) @binding(2)
var imgAbuffer: texture_storage_2d<rgba16float, read_write>;


struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) tex_coord: vec2<f32>,
}


@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOut {

    // creates two vertices that cover the whole screen
    let xy = vec2<f32>(
        f32(in_vertex_index % 2u == 0u),
        f32(in_vertex_index < 2u)
    );
    return VertexOut(vec4<f32>(xy * 2. - (1.), 0., 1.), vec2<f32>(xy.x, 1. - xy.y));
}
@fragment
fn fs_main(vertex: VertexOut) ->  @location(0) vec4<f32> {
    let pixel_loc = vec2<i32>(i32(vertex.pos.x), i32(vertex.pos.y));
    return textureLoad(imgAbuffer, pixel_loc);
}
