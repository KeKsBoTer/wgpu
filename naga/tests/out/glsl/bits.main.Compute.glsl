#version 310 es

precision highp float;
precision highp int;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


void main() {
    int i = 0;
    ivec2 i2_ = ivec2(0);
    ivec3 i3_ = ivec3(0);
    ivec4 i4_ = ivec4(0);
    uint u = 0u;
    uvec2 u2_ = uvec2(0u);
    uvec3 u3_ = uvec3(0u);
    uvec4 u4_ = uvec4(0u);
    vec2 f2_ = vec2(0.0);
    vec4 f4_ = vec4(0.0);
    vec4 _e28 = f4_;
    u = packSnorm4x8(_e28);
    vec4 _e30 = f4_;
    u = packUnorm4x8(_e30);
    vec2 _e32 = f2_;
    u = packSnorm2x16(_e32);
    vec2 _e34 = f2_;
    u = packUnorm2x16(_e34);
    vec2 _e36 = f2_;
    u = packHalf2x16(_e36);
    ivec4 _e38 = i4_;
    u = uint((_e38[0] & 0xFF) | ((_e38[1] & 0xFF) << 8) | ((_e38[2] & 0xFF) << 16) | ((_e38[3] & 0xFF) << 24));
    uvec4 _e40 = u4_;
    u = (_e40[0] & 0xFFu) | ((_e40[1] & 0xFFu) << 8) | ((_e40[2] & 0xFFu) << 16) | ((_e40[3] & 0xFFu) << 24);
    uint _e42 = u;
    f4_ = unpackSnorm4x8(_e42);
    uint _e44 = u;
    f4_ = unpackUnorm4x8(_e44);
    uint _e46 = u;
    f2_ = unpackSnorm2x16(_e46);
    uint _e48 = u;
    f2_ = unpackUnorm2x16(_e48);
    uint _e50 = u;
    f2_ = unpackHalf2x16(_e50);
    uint _e52 = u;
    i4_ = ivec4(bitfieldExtract(int(_e52), 0, 8), bitfieldExtract(int(_e52), 8, 8), bitfieldExtract(int(_e52), 16, 8), bitfieldExtract(int(_e52), 24, 8));
    uint _e54 = u;
    u4_ = uvec4(bitfieldExtract(_e54, 0, 8), bitfieldExtract(_e54, 8, 8), bitfieldExtract(_e54, 16, 8), bitfieldExtract(_e54, 24, 8));
    int _e56 = i;
    int _e57 = i;
    i = bitfieldInsert(_e56, _e57, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    ivec2 _e61 = i2_;
    ivec2 _e62 = i2_;
    i2_ = bitfieldInsert(_e61, _e62, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    ivec3 _e66 = i3_;
    ivec3 _e67 = i3_;
    i3_ = bitfieldInsert(_e66, _e67, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    ivec4 _e71 = i4_;
    ivec4 _e72 = i4_;
    i4_ = bitfieldInsert(_e71, _e72, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uint _e76 = u;
    uint _e77 = u;
    u = bitfieldInsert(_e76, _e77, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uvec2 _e81 = u2_;
    uvec2 _e82 = u2_;
    u2_ = bitfieldInsert(_e81, _e82, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uvec3 _e86 = u3_;
    uvec3 _e87 = u3_;
    u3_ = bitfieldInsert(_e86, _e87, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uvec4 _e91 = u4_;
    uvec4 _e92 = u4_;
    u4_ = bitfieldInsert(_e91, _e92, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    int _e96 = i;
    i = bitfieldExtract(_e96, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    ivec2 _e100 = i2_;
    i2_ = bitfieldExtract(_e100, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    ivec3 _e104 = i3_;
    i3_ = bitfieldExtract(_e104, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    ivec4 _e108 = i4_;
    i4_ = bitfieldExtract(_e108, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uint _e112 = u;
    u = bitfieldExtract(_e112, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uvec2 _e116 = u2_;
    u2_ = bitfieldExtract(_e116, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uvec3 _e120 = u3_;
    u3_ = bitfieldExtract(_e120, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    uvec4 _e124 = u4_;
    u4_ = bitfieldExtract(_e124, int(min(5u, 32u)), int(min(10u, 32u - min(5u, 32u))));
    int _e128 = i;
    i = findLSB(_e128);
    uvec2 _e130 = u2_;
    u2_ = uvec2(findLSB(_e130));
    ivec3 _e132 = i3_;
    i3_ = findMSB(_e132);
    uvec3 _e134 = u3_;
    u3_ = uvec3(findMSB(_e134));
    int _e136 = i;
    i = findMSB(_e136);
    uint _e138 = u;
    u = uint(findMSB(_e138));
    int _e140 = i;
    i = bitCount(_e140);
    ivec2 _e142 = i2_;
    i2_ = bitCount(_e142);
    ivec3 _e144 = i3_;
    i3_ = bitCount(_e144);
    ivec4 _e146 = i4_;
    i4_ = bitCount(_e146);
    uint _e148 = u;
    u = uint(bitCount(_e148));
    uvec2 _e150 = u2_;
    u2_ = uvec2(bitCount(_e150));
    uvec3 _e152 = u3_;
    u3_ = uvec3(bitCount(_e152));
    uvec4 _e154 = u4_;
    u4_ = uvec4(bitCount(_e154));
    int _e156 = i;
    i = bitfieldReverse(_e156);
    ivec2 _e158 = i2_;
    i2_ = bitfieldReverse(_e158);
    ivec3 _e160 = i3_;
    i3_ = bitfieldReverse(_e160);
    ivec4 _e162 = i4_;
    i4_ = bitfieldReverse(_e162);
    uint _e164 = u;
    u = bitfieldReverse(_e164);
    uvec2 _e166 = u2_;
    u2_ = bitfieldReverse(_e166);
    uvec3 _e168 = u3_;
    u3_ = bitfieldReverse(_e168);
    uvec4 _e170 = u4_;
    u4_ = bitfieldReverse(_e170);
}

