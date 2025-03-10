struct VertexOutput {
    @location(0) tex_coord: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> transform: mat4x4<f32>;

@vertex
fn vs_main(
    @location(0) position: vec4<f32>,
    @location(1) tex_coord: vec2<f32>,
) -> VertexOutput {
    var result: VertexOutput;
    result.tex_coord = tex_coord;
    result.position = transform * position;
    return result;
}

@group(0) @binding(1)
var r_color: texture_2d<u32>;
@group(0) @binding(2)
var imgAbuffer: texture_storage_2d<rgba16float, read_write>;


@fragment
fn fs_main(vertex: VertexOutput) ->  @location(0) vec4<f32> {
    let tex = textureLoad(r_color, vec2<i32>(vertex.tex_coord * 256.0), 0);
    let v = f32(tex.x) / 255.0;


    let color = vec4<f32>(1.0 - (v * 5.0), 1.0 - (v * 15.0), 1.0 - (v * 50.0), 0.25);

    let pixel_loc = vec2<i32>(i32(vertex.position.x), i32(vertex.position.y));

    fragmentBarrierBegin();
    
    let old_color = textureLoad(imgAbuffer, pixel_loc);
    let new_color = old_color.rgb + (1.0 - old_color.a) * color.rgb * color.a;
    let new_alpha = old_color.a + (1.0 - old_color.a) * color.a;
    
    textureStore(imgAbuffer, pixel_loc, vec4<f32>(new_color, new_alpha));
    fragmentBarrierEnd();
    return vec4<f32>(new_color, new_alpha);
}
