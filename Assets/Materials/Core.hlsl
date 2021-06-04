#ifndef UNIVERSAL_PIPELINE_CORE_INCLUDED
#define UNIVERSAL_PIPELINE_CORE_INCLUDED

#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Common.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Packing.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Version.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Input.hlsl"

#if !defined(SHADER_HINT_NICE_QUALITY)
#if defined(SHADER_API_MOBILE) || defined(SHADER_API_SWITCH)
#define SHADER_HINT_NICE_QUALITY 0
#else
#define SHADER_HINT_NICE_QUALITY 1
#endif
#endif

// Shader Quality Tiers in Universal.
// SRP doesn't use Graphics Settings Quality Tiers.
// We should expose shader quality tiers in the pipeline asset.
// Meanwhile, it's forced to be:
// High Quality: Non-mobile platforms or shader explicit defined SHADER_HINT_NICE_QUALITY
// Medium: Mobile aside from GLES2
// Low: GLES2
#if SHADER_HINT_NICE_QUALITY
#define SHADER_QUALITY_HIGH
#elif defined(SHADER_API_GLES)
#define SHADER_QUALITY_LOW
#else
#define SHADER_QUALITY_MEDIUM
#endif

#ifndef BUMP_SCALE_NOT_SUPPORTED
#define BUMP_SCALE_NOT_SUPPORTED !SHADER_HINT_NICE_QUALITY
#endif

struct VertexPositionInputs
{
    float3 positionWS; // World space position
    float3 positionVS; // View space position
    float4 positionCS; // Homogeneous clip space position
    float4 positionNDC;// Homogeneous normalized device coordinates
};

struct VertexNormalInputs
{
    real3 tangentWS;
    real3 bitangentWS;
    float3 normalWS;
};

float4x4 ApplyCameraTranslationToMatrix0(float4x4 modelMatrix)
{
    // To handle camera relative rendering we substract the camera position in the model matrix
#if (SHADEROPTIONS_CAMERA_RELATIVE_RENDERING != 0)
    modelMatrix._m03_m13_m23 -= _WorldSpaceCameraPos;
#endif
    return modelMatrix;
}

float4x4 ApplyCameraTranslationToInverseMatrix0(float4x4 inverseModelMatrix)
{
#if (SHADEROPTIONS_CAMERA_RELATIVE_RENDERING != 0)
    // To handle camera relative rendering we need to apply translation before converting to object space
    float4x4 translationMatrix = { { 1.0, 0.0, 0.0, _WorldSpaceCameraPos.x },{ 0.0, 1.0, 0.0, _WorldSpaceCameraPos.y },{ 0.0, 0.0, 1.0, _WorldSpaceCameraPos.z },{ 0.0, 0.0, 0.0, 1.0 } };
    return mul(inverseModelMatrix, translationMatrix);
#else
    return inverseModelMatrix;
#endif
}


float4x4 GetObjectToWorldMatrix0()
{
    return ApplyCameraTranslationToMatrix0(UNITY_MATRIX_M);
}

float4x4 GetWorldToObjectMatrix0()
{
    return ApplyCameraTranslationToInverseMatrix0(UNITY_MATRIX_I_M);
}


float4x4 GetWorldToViewMatrix0()
{
#if (SHADEROPTIONS_CAMERA_RELATIVE_RENDERING != 0)
    float4x4 viewMatrix = UNITY_MATRIX_V;
    viewMatrix._m03_m13_m23 += _WorldSpaceCameraPos * viewMatrix._m00_m11_m22;
    return viewMatrix;
#else
    return UNITY_MATRIX_V;
#endif
}


float3 TransformObjectToWorld0(float3 positionOS)
{
    return mul(GetObjectToWorldMatrix0(), float4(positionOS, 1.0)).xyz;
}

float3 TransformWorldToObject0(float3 positionWS)
{
    return mul(GetWorldToObjectMatrix0(), float4(positionWS, 1.0)).xyz;
}

float3 TransformWorldToView0(float3 positionWS)
{
    return mul(GetWorldToViewMatrix0(), float4(positionWS, 1.0)).xyz;
}


//修改顶点位置到相机相对空间
VertexPositionInputs GetVertexPositionInputs(float3 positionOS)
{
    VertexPositionInputs input;

    input.positionWS = TransformObjectToWorld0(positionOS);
    input.positionVS = TransformWorldToView0(input.positionWS);
    input.positionCS = TransformWViewToHClip(input.positionVS);

    float4 ndc = input.positionCS * 0.5f;
    input.positionNDC.xy = float2(ndc.x, ndc.y * _ProjectionParams.x) + ndc.w;
    input.positionNDC.zw = input.positionCS.zw;

    return input;
}

VertexNormalInputs GetVertexNormalInputs(float3 normalOS)
{
    VertexNormalInputs tbn;
    tbn.tangentWS = real3(1.0, 0.0, 0.0);
    tbn.bitangentWS = real3(0.0, 1.0, 0.0);
    tbn.normalWS = TransformObjectToWorldNormal(normalOS);
    return tbn;
}


float3 TransformObjectToWorldNormal0(float3 normalOS, bool doNormalize = true)
{
#ifdef UNITY_ASSUME_UNIFORM_SCALING
    return TransformObjectToWorldDir(normalOS, doNormalize);
#else
    // Normal need to be multiply by inverse transpose
    float3 normalWS = mul(normalOS, (float3x3)GetWorldToObjectMatrix0());
    if (doNormalize)
        return SafeNormalize(normalWS);

    return normalWS;
#endif
}

float3 TransformObjectToWorldDir0(float3 dirOS, bool doNormalize = true)
{
    float3 dirWS = mul((float3x3)GetObjectToWorldMatrix0(), dirOS);
    if (doNormalize)
        return SafeNormalize(dirWS);

    return dirWS;
}



//修改法线到相机相对空间
VertexNormalInputs GetVertexNormalInputs(float3 normalOS, float4 tangentOS)
{
    VertexNormalInputs tbn;

    // mikkts space compliant. only normalize when extracting normal at frag.
    real sign = tangentOS.w * GetOddNegativeScale();

    tbn.normalWS = TransformObjectToWorldNormal0(normalOS);
    tbn.tangentWS = TransformObjectToWorldDir0(tangentOS.xyz);
    tbn.bitangentWS = cross(tbn.normalWS, tbn.tangentWS) * sign;
    return tbn;
}

#if UNITY_REVERSED_Z
    #if SHADER_API_OPENGL || SHADER_API_GLES || SHADER_API_GLES3
        //GL with reversed z => z clip range is [near, -far] -> should remap in theory but dont do it in practice to save some perf (range is close enough)
        #define UNITY_Z_0_FAR_FROM_CLIPSPACE(coord) max(-(coord), 0)
    #else
        //D3d with reversed Z => z clip range is [near, 0] -> remapping to [0, far]
        //max is required to protect ourselves from near plane not being correct/meaningfull in case of oblique matrices.
        #define UNITY_Z_0_FAR_FROM_CLIPSPACE(coord) max(((1.0-(coord)/_ProjectionParams.y)*_ProjectionParams.z),0)
    #endif
#elif UNITY_UV_STARTS_AT_TOP
    //D3d without reversed z => z clip range is [0, far] -> nothing to do
    #define UNITY_Z_0_FAR_FROM_CLIPSPACE(coord) (coord)
#else
    //Opengl => z clip range is [-near, far] -> should remap in theory but dont do it in practice to save some perf (range is close enough)
    #define UNITY_Z_0_FAR_FROM_CLIPSPACE(coord) (coord)
#endif

float3 GetCameraPositionWS()
{
#if (SHADEROPTIONS_CAMERA_RELATIVE_RENDERING != 0)
    return float3(0, 0, 0);
#else
    return _WorldSpaceCameraPos;
#endif
   
}

float4 GetScaledScreenParams()
{
    return _ScaledScreenParams;
}

void AlphaDiscard(real alpha, real cutoff, real offset = 0.0h)
{
#ifdef _ALPHATEST_ON
    clip(alpha - cutoff + offset);
#endif
}

half OutputAlpha(half outputAlpha, half surfaceType = 0.0)
{
    return surfaceType == 1 ? outputAlpha : 1.0;
}

// A word on normalization of normals:
// For better quality normals should be normalized before and after
// interpolation.
// 1) In vertex, skinning or blend shapes might vary significantly the lenght of normal.
// 2) In fragment, because even outputting unit-length normals interpolation can make it non-unit.
// 3) In fragment when using normal map, because mikktspace sets up non orthonormal basis.
// However we will try to balance performance vs quality here as also let users configure that as
// shader quality tiers.
// Low Quality Tier: Normalize either per-vertex or per-pixel depending if normalmap is sampled.
// Medium Quality Tier: Always normalize per-vertex. Normalize per-pixel only if using normal map
// High Quality Tier: Normalize in both vertex and pixel shaders.
real3 NormalizeNormalPerVertex(real3 normalWS)
{
#if defined(SHADER_QUALITY_LOW) && defined(_NORMALMAP)
    return normalWS;
#else
    return normalize(normalWS);
#endif
}

real3 NormalizeNormalPerPixel(real3 normalWS)
{
#if defined(SHADER_QUALITY_HIGH) || defined(_NORMALMAP)
    return normalize(normalWS);
#else
    return normalWS;
#endif
}

// TODO: A similar function should be already available in SRP lib on master. Use that instead
float4 ComputeScreenPos(float4 positionCS)
{
    float4 o = positionCS * 0.5f;
    o.xy = float2(o.x, o.y * _ProjectionParams.x) + o.w;
    o.zw = positionCS.zw;
    return o;
}

real ComputeFogFactor(float z)
{
    float clipZ_01 = UNITY_Z_0_FAR_FROM_CLIPSPACE(z);

#if defined(FOG_LINEAR)
    // factor = (end-z)/(end-start) = z * (-1/(end-start)) + (end/(end-start))
    float fogFactor = saturate(clipZ_01 * unity_FogParams.z + unity_FogParams.w);
    return real(fogFactor);
#elif defined(FOG_EXP) || defined(FOG_EXP2)
    // factor = exp(-(density*z)^2)
    // -density * z computed at vertex
    return real(unity_FogParams.x * clipZ_01);
#else
    return 0.0h;
#endif
}

real ComputeFogIntensity(real fogFactor)
{
    real fogIntensity = 0.0h;
#if defined(FOG_LINEAR) || defined(FOG_EXP) || defined(FOG_EXP2)
#if defined(FOG_EXP)
    // factor = exp(-density*z)
    // fogFactor = density*z compute at vertex
    fogIntensity = saturate(exp2(-fogFactor));
#elif defined(FOG_EXP2)
    // factor = exp(-(density*z)^2)
    // fogFactor = density*z compute at vertex
    fogIntensity = saturate(exp2(-fogFactor * fogFactor));
#elif defined(FOG_LINEAR)
    fogIntensity = fogFactor;
#endif
#endif
    return fogIntensity;
}

half3 MixFogColor(real3 fragColor, real3 fogColor, real fogFactor)
{
#if defined(FOG_LINEAR) || defined(FOG_EXP) || defined(FOG_EXP2)
    real fogIntensity = ComputeFogIntensity(fogFactor);
    fragColor = lerp(fogColor, fragColor, fogIntensity);
#endif
    return fragColor;
}

half3 MixFog(real3 fragColor, real fogFactor)
{
    return MixFogColor(fragColor, unity_FogColor.rgb, fogFactor);
}

// Stereo-related bits
#if defined(UNITY_STEREO_INSTANCING_ENABLED) || defined(UNITY_STEREO_MULTIVIEW_ENABLED)

    #define SLICE_ARRAY_INDEX   unity_StereoEyeIndex

    #define TEXTURE2D_X(textureName)                                        TEXTURE2D_ARRAY(textureName)
    #define TEXTURE2D_X_PARAM(textureName, samplerName)                     TEXTURE2D_ARRAY_PARAM(textureName, samplerName)
    #define TEXTURE2D_X_ARGS(textureName, samplerName)                      TEXTURE2D_ARRAY_ARGS(textureName, samplerName)
    #define TEXTURE2D_X_HALF(textureName)                                   TEXTURE2D_ARRAY_HALF(textureName)
    #define TEXTURE2D_X_FLOAT(textureName)                                  TEXTURE2D_ARRAY_FLOAT(textureName)

    #define LOAD_TEXTURE2D_X(textureName, unCoord2)                         LOAD_TEXTURE2D_ARRAY(textureName, unCoord2, SLICE_ARRAY_INDEX)
    #define LOAD_TEXTURE2D_X_LOD(textureName, unCoord2, lod)                LOAD_TEXTURE2D_ARRAY_LOD(textureName, unCoord2, SLICE_ARRAY_INDEX, lod)
    #define SAMPLE_TEXTURE2D_X(textureName, samplerName, coord2)            SAMPLE_TEXTURE2D_ARRAY(textureName, samplerName, coord2, SLICE_ARRAY_INDEX)
    #define SAMPLE_TEXTURE2D_X_LOD(textureName, samplerName, coord2, lod)   SAMPLE_TEXTURE2D_ARRAY_LOD(textureName, samplerName, coord2, SLICE_ARRAY_INDEX, lod)
    #define GATHER_TEXTURE2D_X(textureName, samplerName, coord2)            GATHER_TEXTURE2D_ARRAY(textureName, samplerName, coord2, SLICE_ARRAY_INDEX)
    #define GATHER_RED_TEXTURE2D_X(textureName, samplerName, coord2)        GATHER_RED_TEXTURE2D(textureName, samplerName, float3(coord2, SLICE_ARRAY_INDEX))
    #define GATHER_GREEN_TEXTURE2D_X(textureName, samplerName, coord2)      GATHER_GREEN_TEXTURE2D(textureName, samplerName, float3(coord2, SLICE_ARRAY_INDEX))
    #define GATHER_BLUE_TEXTURE2D_X(textureName, samplerName, coord2)       GATHER_BLUE_TEXTURE2D(textureName, samplerName, float3(coord2, SLICE_ARRAY_INDEX))

#else

    #define SLICE_ARRAY_INDEX       0

    #define TEXTURE2D_X(textureName)                                        TEXTURE2D(textureName)
    #define TEXTURE2D_X_PARAM(textureName, samplerName)                     TEXTURE2D_PARAM(textureName, samplerName)
    #define TEXTURE2D_X_ARGS(textureName, samplerName)                      TEXTURE2D_ARGS(textureName, samplerName)
    #define TEXTURE2D_X_HALF(textureName)                                   TEXTURE2D_HALF(textureName)
    #define TEXTURE2D_X_FLOAT(textureName)                                  TEXTURE2D_FLOAT(textureName)

    #define LOAD_TEXTURE2D_X(textureName, unCoord2)                         LOAD_TEXTURE2D(textureName, unCoord2)
    #define LOAD_TEXTURE2D_X_LOD(textureName, unCoord2, lod)                LOAD_TEXTURE2D_LOD(textureName, unCoord2, lod)
    #define SAMPLE_TEXTURE2D_X(textureName, samplerName, coord2)            SAMPLE_TEXTURE2D(textureName, samplerName, coord2)
    #define SAMPLE_TEXTURE2D_X_LOD(textureName, samplerName, coord2, lod)   SAMPLE_TEXTURE2D_LOD(textureName, samplerName, coord2, lod)
    #define GATHER_TEXTURE2D_X(textureName, samplerName, coord2)            GATHER_TEXTURE2D(textureName, samplerName, coord2)
    #define GATHER_RED_TEXTURE2D_X(textureName, samplerName, coord2)        GATHER_RED_TEXTURE2D(textureName, samplerName, coord2)
    #define GATHER_GREEN_TEXTURE2D_X(textureName, samplerName, coord2)      GATHER_GREEN_TEXTURE2D(textureName, samplerName, coord2)
    #define GATHER_BLUE_TEXTURE2D_X(textureName, samplerName, coord2)       GATHER_BLUE_TEXTURE2D(textureName, samplerName, coord2)

#endif

#if defined(UNITY_SINGLE_PASS_STEREO)
float2 TransformStereoScreenSpaceTex(float2 uv, float w)
{
    // TODO: RVS support can be added here, if Universal decides to support it
    float4 scaleOffset = unity_StereoScaleOffset[unity_StereoEyeIndex];
    return uv.xy * scaleOffset.xy + scaleOffset.zw * w;
}

float2 UnityStereoTransformScreenSpaceTex(float2 uv)
{
    return TransformStereoScreenSpaceTex(saturate(uv), 1.0);
}

#else

#define UnityStereoTransformScreenSpaceTex(uv) uv

#endif // defined(UNITY_SINGLE_PASS_STEREO)

#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Deprecated.hlsl"

#endif
