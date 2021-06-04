### 启动camera relative rendering
   选择 Materials/Lit.shader
   第一个pass
   #define SHADEROPTIONS_CAMERA_RELATIVE_RENDERING (1)
### 主要修改文件 Materials/Core.hlsl
   #if(SHADEROPTIONS_CAMERA_RELATIVE_RENDERING!=0)都是添加的；主要涉及相机位置，model矩阵运算，view矩阵运算