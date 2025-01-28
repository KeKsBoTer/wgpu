struct Data {
    vecs: array<vec4<f32>, 42>,
}

const NUM_VECS: i32 = 42i;

@group(1) @binding(0) 
var<uniform> global: Data;

fn function() -> vec4<f32> {
    var sum: vec4<f32> = vec4(0f);
    var i: i32 = 0i;

    loop {
        let _e9 = i;
        if !((_e9 < NUM_VECS)) {
            break;
        }
        {
            let _e15 = sum;
            let _e16 = i;
            let _e18 = global.vecs[_e16];
            sum = (_e15 + _e18);
        }
        continuing {
            let _e12 = i;
            i = (_e12 + 1i);
        }
    }
    let _e20 = sum;
    return _e20;
}

fn main_1() {
}

@fragment 
fn main() {
    main_1();
    return;
}
