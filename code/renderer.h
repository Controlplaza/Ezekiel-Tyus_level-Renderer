// minimalistic code to draw a single triangle, this is not part of the API.
// TODO: Part 1b
#include"FSLogo.h"
#include "shaderc/shaderc.h" // needed for compiling shaders at runtime
#include <chrono>
#include <fstream>
#include <iostream>
#include "../code/h2bParser.h"

#ifdef _WIN32 // must use MT platform DLL libraries on windows
#pragma comment(lib, "shaderc_combined.lib") 
#endif

// Simple Vertex Shader
const char* vertexShaderSource = R"(
// TODO: 2i

#pragma pack_matrix(row_major)
#define MAX_SUBMESH_PER_DRAW 1024


// an ultra simple hlsl vertex shader
// TODO: Part 2b
struct OBJ_ATTRIBUTES
{
    float3 Kd; // diffuse reflectivity
    float1 d; // dissolve (transparency) 
    float3 Ks; // specular reflectivity
    float1 Ns; // specular exponent
    float3 Ka; // ambient reflectivity
    float1 sharpness; // local reflection map sharpness
    float3 Tf; // transmission filter
    float1 Ni; // optical density (index of refraction)
    float3 Ke; // emissive reflectivity
    uint illum; // illumination model
};
struct SHADER_data
{
    float4 LightDir;
    float4 LightColor;
    matrix view;
    matrix projection;
    matrix matrices[MAX_SUBMESH_PER_DRAW];
    OBJ_ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];
 
    float4 cameraPos;
    float4 ambientlight;
};

StructuredBuffer<SHADER_data> scenedata;
[[vk::push_constant]] 
cbuffer MESHINDEX
{

    float4x4 meshMatrix;
    uint meshID;
};
// TODO: Part 4g
// TODO: Part 2i

// TODO: Part 3e
// TODO: Part 4a
//struct OUTPUT_TO_RASTERIZER
//{
//    float4 posH : SV_Position;
//    float3 nrmW : nrm;  
//};
// TODO: Part 1f
struct VS_INPUT
{
    float3 pos : POSITION;
    float3 uvw : UVW;
    float3 nrm : NORMAL;

};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 nrm : NORMAL;
    float4 Posw : POSITION;
};


// TODO: Part 4b
VS_OUTPUT main(VS_INPUT input)
{
 
    VS_OUTPUT griff;

    griff.pos = float4(input.pos, 1);
    griff.nrm = float4(input.nrm, 0);


    griff.pos = mul(griff.pos, meshMatrix);
    griff.Posw = griff.pos;
    griff.pos = mul(griff.pos, scenedata[0].view);
    griff.pos = mul(griff.pos, scenedata[0].projection);

    griff.nrm = mul(griff.pos, scenedata[0].matrices[meshID]);

    return griff;

}
)";
// Simple Pixel Shader
const char* pixelShaderSource = R"(


#define MAX_SUBMESH_PER_DRAW 1024


// an ultra simple hlsl vertex shader
// TODO: Part 2b
struct OBJ_ATTRIBUTES
{
    float3 Kd; // diffuse reflectivity
    float1 d; // dissolve (transparency) 
    float3 Ks; // specular reflectivity
    float1 Ns; // specular exponent
    float3 Ka; // ambient reflectivity
    float1 sharpness; // local reflection map sharpness
    float3 Tf; // transmission filter
    float1 Ni; // optical density (index of refraction)
    float3 Ke; // emissive reflectivity
    uint illum; // illumination model
};
struct SHADER_data
{
    float4 LightDir;
    float4 LightColor;
    matrix view;
    matrix projection;
    matrix matrices[MAX_SUBMESH_PER_DRAW];
    OBJ_ATTRIBUTES materials[MAX_SUBMESH_PER_DRAW];
 
    float4 cameraPos;
    float4 ambientlight;
};

StructuredBuffer<SHADER_data> scenedata;
[[vk::push_constant]] 
cbuffer MESHINDEX
{
    float4x4 meshMatrix;
    uint meshID;
};
// TODO: Part 4g
// TODO: Part 2i

// TODO: Part 3e
// TODO: Part 4a
//struct OUTPUT_TO_RASTERIZER
//{
//    float4 posH : SV_Position;
//    float3 nrmW : nrm;  
//};
// TODO: Part 1f
struct VS_INPUT
{
    float3 pos : POSITION;
    float3 uvw : UVW;
    float3 nrm : NORMAL;

};

struct VS_OUTPUT
{
    float4 pos : SV_POSITION;
    float4 nrm : NORMAL;
    float4 Posw : POSITION;
};
float4 main(VS_OUTPUT inputverty) : SV_TARGET
{
    float3 lightdirection = normalize(scenedata[0].LightDir.xyz);
    float4 lightcolor = scenedata[0].LightColor;
    float3 surfacepos = inputverty.pos.xyz;
    float3 surfacenormal = normalize(inputverty.nrm.xyz);
    float4 surfacecolor = float4(scenedata[0].materials[meshID].Kd, 1);


    float4 finalcolor;
    float lightratio = clamp(dot(-lightdirection, surfacenormal), 0, 1);
    float4 ambient = scenedata[0].ambientlight * surfacecolor;


    finalcolor = lightratio * scenedata[0].LightColor * surfacecolor;

    finalcolor += ambient;




    float3 reflect = reflect(lightdirection, surfacenormal);
    float3 tocam = normalize(scenedata[0].cameraPos.xyz - inputverty.Posw.xyz);
    float specdot = dot(reflect, tocam);
    specdot = pow(specdot, scenedata[0].materials[meshID].Ns * 0.5f);
    float4 specfinalcolor = float4(1.0f, 1.0f, 1.0f, 0.0f) * lightcolor * saturate(specdot);

    finalcolor += specfinalcolor;
    return finalcolor;

}
)";

class LevelRenderer
{
public:

	VkBuffer IndexHandle = nullptr;
	VkDeviceMemory IndexData = nullptr;
	VkDevice Device = nullptr;
	VkBuffer VertexHandle = nullptr;
	VkDeviceMemory VertexData = nullptr;
	H2B::VERTEX* ModoVerts;
	int* logoIndices;
	unsigned vertexCount;
	unsigned indexCount;
	unsigned materialCount;
	std::vector<H2B::VERTEX> vertices;
	std::vector<unsigned > indices;
	std::vector<H2B::MATERIAL> materials;
	std::vector<H2B::BATCH> batches;
	std::vector<H2B::MESH> meshes;
	GW::MATH::GMATRIXF MeshMatrix;

	unsigned meshCount;

	LevelRenderer(std::string model)
	{
		//varibales using all throughout function

		H2B::Parser parsy;
		//model variables to parse variables


		std::string meshvar;
		int itemId = model.find_first_of('.');
		meshvar = model.substr(0, itemId);
		//model names
		//std::string modAddress = "C:/Users/ezeki/OneDrive/Desktop/Ezekiel - Tyus_level - Renderer/codelevel stuff";
		std::string modAddress = "../level stuff/";
		std::string Mesh2H2bname = modAddress.append(meshvar);
		Mesh2H2bname = Mesh2H2bname.append(".h2b");
		const char* meshparsnameschar = Mesh2H2bname.c_str();
		//pasring of h2b files
		parsy.Parse(meshparsnameschar);
		vertexCount = parsy.vertexCount;
		indexCount = parsy.indexCount;
		materialCount = parsy.materialCount;
		meshCount = parsy.meshCount;
		vertices = parsy.vertices;
		indices = parsy.indices;
		materials = parsy.materials;
		batches = parsy.batches;
		meshes = parsy.meshes;

		//populating materials
		for (size_t i = 0; i < parsy.materialCount; i++)
		{
			std::cout << Mesh2H2bname << " ";
			materials[i].attrib = parsy.materials[i].attrib;
			//parsy.materials[i].attrib = materials[i].attrib;// puts the materials from h2b parsed info to regular info
		}
		//open file
		std::string line;
		std::fstream myFile;
		myFile.open("../GameLevel.txt", std::ios::in);

		if (myFile.is_open())
		{
			int linecounter = 0;
			while (std::getline(myFile, line))
			{
				std::cout << line << '\n';

				if (line == "MESH")
				{
					while (std::getline(myFile, line))
					{
						if (line == model)
						{
							while (std::getline(myFile, line) && linecounter != 4)
							{


								linecounter++;
								GW::MATH::GMATRIXF linematrix;
								if (linecounter == 1)
								{
									std::vector<std::string> modelcoordinates;
									std::string newline = line.substr(13);
									std::stringstream stream(newline);
									while (stream.good())
									{
										std::string numbers;
										std::getline(stream, numbers, ',');
										modelcoordinates.push_back(numbers);
									}
									linematrix.row1.x = std::stof(modelcoordinates[0]);
									linematrix.row1.y = std::stof(modelcoordinates[1]);
									linematrix.row1.z = std::stof(modelcoordinates[2]);
									linematrix.row1.w = std::stof(modelcoordinates[3]);
								}
								if (linecounter == 2)
								{
									std::vector<std::string> modelcoordinates2;
									std::string newline2 = line.substr(13);
									std::stringstream stream2(newline2);

									while (stream2.good())
									{
										std::string numbers2;
										std::getline(stream2, numbers2, ',');
										modelcoordinates2.push_back(numbers2);
									}
									linematrix.row2.x = std::stof(modelcoordinates2[0]);
									linematrix.row2.y = std::stof(modelcoordinates2[1]);
									linematrix.row2.z = std::stof(modelcoordinates2[2]);
									linematrix.row2.w = std::stof(modelcoordinates2[3]);
								}
								if (linecounter == 3)
								{
									std::vector<std::string> modelcoordinates3;
									std::string newline3 = line.substr(13);
									std::stringstream stream3(newline3);

									while (stream3.good())
									{
										std::string numbers3;
										std::getline(stream3, numbers3, ',');
										modelcoordinates3.push_back(numbers3);
									}
									linematrix.row3.x = std::stof(modelcoordinates3[0]);
									linematrix.row3.y = std::stof(modelcoordinates3[1]);
									linematrix.row3.z = std::stof(modelcoordinates3[2]);
									linematrix.row3.w = std::stof(modelcoordinates3[3]);
								}

								if (linecounter == 4)
								{
									std::vector<std::string> modelcoordinates4;
									std::string newline4 = line.substr(13);
									std::stringstream stream4(newline4);
									while (stream4.good())
									{
										std::string numbers4;
										std::getline(stream4, numbers4, ',');
										modelcoordinates4.push_back(numbers4);
									}
									linematrix.row4.x = std::stof(modelcoordinates4[0]);
									linematrix.row4.y = std::stof(modelcoordinates4[1]);
									linematrix.row4.z = std::stof(modelcoordinates4[2]);
									linematrix.row4.w = std::stof(modelcoordinates4[3]);
									MeshMatrix = linematrix;
									linecounter = 0;
									break;
								}
							}
						}
						else { break; }

					}
				}
			}
			myFile.close();
		}
		ModoVerts = new H2B::VERTEX[vertexCount];
		for (size_t i = 0; i < vertexCount; i++)
		{
			ModoVerts[i] = vertices[i];
		}
		logoIndices = new int[indexCount];
		for (size_t i = 0; i < indexCount; i++)
		{
			logoIndices[i] = indices[i];
		}
	};
};

// Creation, Rendering & Cleanup
class Renderer
{


	//light
	struct Light
	{
		float coneRatio;
		float LightColor[4];
		float lightDirection[3];
		float totalDeltaTime[2];
		float lightPosition[3];
		float lightFalloff[2];
	};
	// TODO: Part 2b
#define MAX_SUBMESH_PER_DRAW 1024
	struct SHADER_Data
	{
		//view data

		GW::MATH::GVECTORF LightDir;
		GW::MATH::GVECTORF LightColor;
		GW::MATH::GMATRIXF View;
		GW::MATH::GMATRIXF Projection;
		//lighting data
		GW::MATH::GMATRIXF matrices[MAX_SUBMESH_PER_DRAW];
		H2B::ATTRIBUTES matierials[MAX_SUBMESH_PER_DRAW];

		GW::MATH::GVECTORF camPos;
		GW::MATH::GVECTORF ambientLight;


	};
	SHADER_Data model;
	// proxy handles
	GW::SYSTEM::GWindow win;
	GW::GRAPHICS::GVulkanSurface vlk;
	GW::CORE::GEventReceiver shutdown;

	// what we need at a minimum to draw a triangle
	VkDevice device = nullptr;
	VkBuffer vertexHandle = nullptr;
	VkDeviceMemory vertexData = nullptr;
	GW::MATH::GVECTORF LightDir = { -1.0f, -1.0f, 2.0f, 0.0f };
	GW::MATH::GVECTORF LightColor = { 0.9f, 0.9, 1.0f, 1.0f };

	// TODO: Part 1g
	VkBuffer IndexHandle = nullptr;
	VkDeviceMemory indexData = nullptr;
	// TODO: Part 2c

	std::vector<VkDeviceMemory> storageMemory;
	std::vector<VkDevice> storagedevice;
	std::vector<VkBuffer> storageBuffer;




	VkShaderModule vertexShader = nullptr;
	VkShaderModule pixelShader = nullptr;
	// pipeline settings for drawing (also required)
	VkPipeline pipeline = nullptr;
	VkPipelineLayout pipelineLayout = nullptr;
	// TODO: Part 2e
	VkDescriptorSetLayout descriptorlayout;
	// TODO: Part 2f
	VkDescriptorPool descriptorPool;
	// TODO: Part 2g
	std::vector<VkDescriptorSet>descriptorSet;
	// TODO: Part 4f

// TODO: Part 2a
	GW::MATH::GMATRIXF World;
	GW::MATH::GMATRIXF View;
	GW::MATH::GMATRIXF Projection;
	GW::MATH::GMatrix proxyMan;
	// TODO: Part 2b
	GW::MATH::GMATRIXF matrices[MAX_SUBMESH_PER_DRAW];
	OBJ_ATTRIBUTES matierials[MAX_SUBMESH_PER_DRAW];
	// TODO: Part 4g
	GW::INPUT::GInput playerInput;
	GW::INPUT::GController controllerInput;
	std::vector<LevelRenderer*>OBJMESH;
public:

	struct MeshIndex
	{
		GW::MATH::GMATRIXF meshMatrix;
		unsigned int meshID;
	};
	Renderer(GW::SYSTEM::GWindow _win, GW::GRAPHICS::GVulkanSurface _vlk)
	{
		win = _win;
		vlk = _vlk;
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// TODO: Part 2a

		for (int i = 0; i < FSLogo_vertexcount; i++)
		{
			matrices[i].data;
		}
		playerInput.Create(win);
		proxyMan.Create();
		std::chrono::time_point<std::chrono::system_clock> start, end;
		start = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		GW::MATH::GVECTORF translatevar;
		translatevar.x = 0.75f;	//void render
		translatevar.y = 0.25f;	//void render
		translatevar.z = -1.5f;	//void render
		World = GW::MATH::GIdentityMatrixF;
		View = GW::MATH::GIdentityMatrixF;
		proxyMan.RotateYLocalF(World, elapsed_seconds.count(), World); // void render
		proxyMan.TranslateLocalF(View, translatevar, World); //void render
		float aspectratio = 0.0f;
		vlk.GetAspectRatio(aspectratio);

		proxyMan.ProjectionVulkanLHF(G2D_DEGREE_TO_RADIAN(65.0f), aspectratio, 0.1f, 100.0f, Projection);
		model.Projection = Projection;

		GW::MATH::GVECTORF eye{ 0.75f,0.25f,-1.5f,0.0f };
		GW::MATH::GVECTORF at{ 0.15f,0.75f,0.0f,0.0f };
		GW::MATH::GVECTORF up{ 0.0f,1.0f,0.0,0.0 };
		proxyMan.LookAtLHF(eye, at, up, View);
		model.View = View;
		proxyMan.InverseF(View, View);

		//levelmodel containerss

		std::vector<GW::MATH::GMATRIXF> matrixcounter;
		std::vector<unsigned int> materialID;
		std::vector<H2B::ATTRIBUTES> Model_Materials;
		//std::cout << "view is at " << View.row4.x << " " << View.row4.y << " " << View.row4.z;

		float LightVectLength = std::sqrt(
			(LightDir.x * LightDir.x) +	  //dot product
			(LightDir.y * LightDir.y) +	  //dot product
			(LightDir.z * LightDir.z));	  //dot product
		LightDir.x /= LightVectLength;
		LightDir.y /= LightVectLength;
		LightDir.z /= LightVectLength;
		model.LightColor = LightColor;
		GW::MATH::GVector::NormalizeF(LightDir, LightColor);
		model.LightDir = LightDir;
		model.camPos = View.row1;
		model.ambientLight = { 0.25f,0.25f,8.35f,0.0f };

		std::string fileAddress = "../GameLevel.txt";
		std::string line;
		std::fstream myfile(fileAddress);
		if (myfile.is_open())
		{
			int lineCount = 0;
			while (std::getline(myfile, line))
			{
				if (line == "MESH")
				{
					while (std::getline(myfile, line))
					{
						lineCount++;
						GW::MATH::GMATRIXF objectMaterial;
						if (lineCount == 1)
						{
							std::string meshy;
							meshy += line;
							LevelRenderer* levelinstance = new LevelRenderer(meshy);
							OBJMESH.push_back(levelinstance);
							lineCount = 0;
							break;
						}
					}
				}


			}
			myfile.close();
		}

		//loading meshes
		int MeshCount = 0;
		for (size_t i = 0; i < OBJMESH.size(); i++)
		{
			for (size_t j = 0; j < OBJMESH[i]->meshCount; j++)
			{
				model.matrices[MeshCount] = OBJMESH[i]->MeshMatrix;
				MeshCount++;
			}
		}

		for (size_t i = 0; i < OBJMESH.size(); i++)
		{
			for (size_t j = 0; j < OBJMESH[i]->meshCount; j++)
			{
				Model_Materials.push_back(OBJMESH[i]->materials[j].attrib);
			}
		}

		for (size_t i = 0; i < Model_Materials.size(); i++)
		{
			model.matierials[i] = Model_Materials[i];
		}
		// TODO: Part 2b
		// TODO: Part 4g
		// TODO: part 3b

		/***************** GEOMETRY INTIALIZATION ******************/
		// Grab the device & physical device so we can allocate some stuff
		VkPhysicalDevice physicalDevice = nullptr;
		vlk.GetDevice((void**)&device);
		vlk.GetPhysicalDevice((void**)&physicalDevice);


		for (size_t i = 0; i < OBJMESH.size(); i++)
		{
			GvkHelper::create_buffer(physicalDevice, device, sizeof(H2B::VERTEX) * OBJMESH[i]->vertexCount,
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &OBJMESH[i]->VertexHandle, &OBJMESH[i]->VertexData);
			GvkHelper::write_to_buffer(device, OBJMESH[i]->VertexData, OBJMESH[i]->ModoVerts, sizeof(H2B::VERTEX) * OBJMESH[i]->vertexCount);


			GvkHelper::create_buffer(physicalDevice, device, sizeof(int) * OBJMESH[i]->indexCount,
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &OBJMESH[i]->IndexHandle, &OBJMESH[i]->IndexData);
			GvkHelper::write_to_buffer(device, OBJMESH[i]->IndexData, OBJMESH[i]->logoIndices, sizeof(int) * OBJMESH[i]->indexCount);
		}
		// TODO: Part 1c
		// Create Vertex Buffer




		// TODO: Part 2d
		unsigned int maxFrames;
		vlk.GetSwapchainImageCount(maxFrames);
		storageBuffer.resize(maxFrames);
		storageMemory.resize(maxFrames);
		descriptorSet.resize(maxFrames);
		/*for (int i = 0; i < 2 ; i++)
		{

			model.matrices[i] = GW::MATH::GIdentityMatrixF;

		}

		for (int i = 0; i < FSLogo_materialcount; i++)
		{
		model.matierials[i] = FSLogo_materials[i].attrib;

		}*/




		for (size_t i = 0; i < maxFrames; i++)
		{

			GvkHelper::create_buffer(physicalDevice, device, sizeof(model),
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
				VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &storageBuffer[i], &storageMemory[i]);
			GvkHelper::write_to_buffer(device, storageMemory[i], &model, sizeof(model));
		}


		/***************** SHADER INTIALIZATION ******************/
		// Intialize runtime shader compiler HLSL -> SPIRV
		shaderc_compiler_t compiler = shaderc_compiler_initialize();
		shaderc_compile_options_t options = shaderc_compile_options_initialize();
		shaderc_compile_options_set_source_language(options, shaderc_source_language_hlsl);
		shaderc_compile_options_set_invert_y(options, false); // TODO: Part 2i
#ifndef NDEBUG
		shaderc_compile_options_set_generate_debug_info(options);
#endif
		// Create Vertex Shader
		shaderc_compilation_result_t result = shaderc_compile_into_spv( // compile
			compiler, vertexShaderSource, strlen(vertexShaderSource),
			shaderc_vertex_shader, "main.vert", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Vertex Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &vertexShader);
		shaderc_result_release(result); // done
		// Create Pixel Shader
		result = shaderc_compile_into_spv( // compile
			compiler, pixelShaderSource, strlen(pixelShaderSource),
			shaderc_fragment_shader, "main.frag", "main", options);
		if (shaderc_result_get_compilation_status(result) != shaderc_compilation_status_success) // errors?
			std::cout << "Pixel Shader Errors: " << shaderc_result_get_error_message(result) << std::endl;
		GvkHelper::create_shader_module(device, shaderc_result_get_length(result), // load into Vulkan
			(char*)shaderc_result_get_bytes(result), &pixelShader);
		shaderc_result_release(result); // done
		// Free runtime shader compiler resources
		shaderc_compile_options_release(options);
		shaderc_compiler_release(compiler);

		/***************** PIPELINE INTIALIZATION ******************/
		// Create Pipeline & Layout (Thanks Tiny!)
		VkRenderPass renderPass;
		vlk.GetRenderPass((void**)&renderPass);
		VkPipelineShaderStageCreateInfo stage_create_info[2] = {};
		// Create Stage Info for Vertex Shader
		stage_create_info[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stage_create_info[0].module = vertexShader;
		stage_create_info[0].pName = "main";
		// Create Stage Info for Fragment Shader
		stage_create_info[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stage_create_info[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stage_create_info[1].module = pixelShader;
		stage_create_info[1].pName = "main";
		// Assembly State
		VkPipelineInputAssemblyStateCreateInfo assembly_create_info = {};
		assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assembly_create_info.primitiveRestartEnable = false;
		// TODO: Part 1e

		// Vertex Input State
		VkVertexInputBindingDescription vertex_binding_description = {};
		vertex_binding_description.binding = 0;
		vertex_binding_description.stride = sizeof(H2B::VERTEX);
		vertex_binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
		VkVertexInputAttributeDescription vertex_attribute_description[3] = {
			{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0 }, //uv, normal, etc....
			{ 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(H2B::VERTEX,uvw) },
			{ 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(H2B::VERTEX,nrm) }
		};
		VkPipelineVertexInputStateCreateInfo input_vertex_info = {};
		input_vertex_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		input_vertex_info.vertexBindingDescriptionCount = 1;
		input_vertex_info.pVertexBindingDescriptions = &vertex_binding_description;
		input_vertex_info.vertexAttributeDescriptionCount = 3;
		input_vertex_info.pVertexAttributeDescriptions = vertex_attribute_description;
		// Viewport State (we still need to set this up even though we will overwrite the values)
		VkViewport viewport = {
			0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
		};
		VkRect2D scissor = { {0, 0}, {width, height} };
		VkPipelineViewportStateCreateInfo viewport_create_info = {};
		viewport_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_create_info.viewportCount = 1;
		viewport_create_info.pViewports = &viewport;
		viewport_create_info.scissorCount = 1;
		viewport_create_info.pScissors = &scissor;
		// Rasterizer State
		VkPipelineRasterizationStateCreateInfo rasterization_create_info = {};
		rasterization_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterization_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterization_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterization_create_info.lineWidth = 1.0f;
		rasterization_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterization_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterization_create_info.depthClampEnable = VK_FALSE;
		rasterization_create_info.depthBiasEnable = VK_FALSE;
		rasterization_create_info.depthBiasClamp = 0.0f;
		rasterization_create_info.depthBiasConstantFactor = 0.0f;
		rasterization_create_info.depthBiasSlopeFactor = 0.0f;
		// Multisampling State
		VkPipelineMultisampleStateCreateInfo multisample_create_info = {};
		multisample_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisample_create_info.sampleShadingEnable = VK_FALSE;
		multisample_create_info.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		multisample_create_info.minSampleShading = 1.0f;
		multisample_create_info.pSampleMask = VK_NULL_HANDLE;
		multisample_create_info.alphaToCoverageEnable = VK_FALSE;
		multisample_create_info.alphaToOneEnable = VK_FALSE;
		// Depth-Stencil State
		VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = {};
		depth_stencil_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil_create_info.depthTestEnable = VK_TRUE;
		depth_stencil_create_info.depthWriteEnable = VK_TRUE;
		depth_stencil_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_create_info.minDepthBounds = 0.0f;
		depth_stencil_create_info.maxDepthBounds = 1.0f;
		depth_stencil_create_info.stencilTestEnable = VK_FALSE;
		// Color Blending Attachment & State
		VkPipelineColorBlendAttachmentState color_blend_attachment_state = {};
		color_blend_attachment_state.colorWriteMask = 0xF;
		color_blend_attachment_state.blendEnable = VK_FALSE;
		color_blend_attachment_state.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_COLOR;
		color_blend_attachment_state.dstColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
		color_blend_attachment_state.colorBlendOp = VK_BLEND_OP_ADD;
		color_blend_attachment_state.srcAlphaBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		color_blend_attachment_state.dstAlphaBlendFactor = VK_BLEND_FACTOR_DST_ALPHA;
		color_blend_attachment_state.alphaBlendOp = VK_BLEND_OP_ADD;
		VkPipelineColorBlendStateCreateInfo color_blend_create_info = {};
		color_blend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_create_info.logicOpEnable = VK_FALSE;
		color_blend_create_info.logicOp = VK_LOGIC_OP_COPY;
		color_blend_create_info.attachmentCount = 1;
		color_blend_create_info.pAttachments = &color_blend_attachment_state;
		color_blend_create_info.blendConstants[0] = 0.0f;
		color_blend_create_info.blendConstants[1] = 0.0f;
		color_blend_create_info.blendConstants[2] = 0.0f;
		color_blend_create_info.blendConstants[3] = 0.0f;
		// Dynamic State 
		VkDynamicState dynamic_state[2] = {
			// By setting these we do not need to re-create the pipeline on Resize
			VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR
		};
		VkPipelineDynamicStateCreateInfo dynamic_create_info = {};
		dynamic_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_create_info.dynamicStateCount = 2;
		dynamic_create_info.pDynamicStates = dynamic_state;

		// TODO: Part 2e
		VkDescriptorSetLayoutBinding descriptor_layout_binding = {};
		descriptor_layout_binding.binding = 0;
		descriptor_layout_binding.descriptorCount = 1;
		descriptor_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descriptor_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
		descriptor_layout_binding.pImmutableSamplers = nullptr;

		VkDescriptorSetLayoutCreateInfo  descriptor_layout_create_info = {};
		descriptor_layout_create_info.bindingCount = 1;
		descriptor_layout_create_info.flags = 0;

		descriptor_layout_create_info.pBindings = &descriptor_layout_binding;

		descriptor_layout_create_info.pNext = nullptr;
		descriptor_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;

		VkResult r = vkCreateDescriptorSetLayout(device, &descriptor_layout_create_info,
			nullptr, &descriptorlayout);

		// TODO: Part 2f
		VkDescriptorPoolCreateInfo descriptor_pool_create_info = {};
		VkDescriptorPoolSize descriptor_pool_size = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,maxFrames };
		descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptor_pool_create_info.poolSizeCount = 1;

		descriptor_pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
		descriptor_pool_create_info.pPoolSizes = &descriptor_pool_size;
		descriptor_pool_create_info.maxSets = maxFrames;
		descriptor_pool_create_info.pNext = nullptr;

		vkCreateDescriptorPool(device, &descriptor_pool_create_info, nullptr, &descriptorPool);

		// TODO: Part 2g
		VkDescriptorSetAllocateInfo descriptor_set_allocate_info = {};
		descriptor_set_allocate_info.descriptorPool = descriptorPool;
		descriptor_set_allocate_info.descriptorSetCount = 1;
		descriptor_set_allocate_info.pNext = nullptr;
		descriptor_set_allocate_info.pSetLayouts = &descriptorlayout;
		descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;

		for (int i = 0; i < maxFrames; ++i)
		{
			r = vkAllocateDescriptorSets(device, &descriptor_set_allocate_info, &descriptorSet[i]);
		}
		// TODO: Part 4f
		// TODO: Part 4f
	// TODO: Part 2h
		for (int i = 0; i < maxFrames; ++i)
		{
			VkDescriptorBufferInfo BufferInfo = { storageBuffer[i],0,VK_WHOLE_SIZE };
			VkWriteDescriptorSet writeToset = {};
			writeToset.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			writeToset.descriptorCount = 1;
			writeToset.dstArrayElement = 0;
			writeToset.dstBinding = 0;
			writeToset.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
			writeToset.dstSet = descriptorSet[i];
			writeToset.pBufferInfo = &BufferInfo;
			vkUpdateDescriptorSets(device, 1, &writeToset, 0, nullptr);
		}
		// TODO: Part 4f
	// TODO: Part 2e

	// Descriptor pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.pSetLayouts = &descriptorlayout;
		pipeline_layout_create_info.setLayoutCount = 1;


		VkPushConstantRange push_constant_range = {};
		push_constant_range.offset = 0;
		push_constant_range.size = sizeof(MeshIndex);
		push_constant_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		// TODO: Part 3c
		pipeline_layout_create_info.pushConstantRangeCount = 1;
		pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;
		vkCreatePipelineLayout(device, &pipeline_layout_create_info,
			nullptr, &pipelineLayout);
		// Pipeline State... (FINALLY) 
		VkGraphicsPipelineCreateInfo pipeline_create_info = {};
		pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeline_create_info.stageCount = 2;
		pipeline_create_info.pStages = stage_create_info;
		pipeline_create_info.pInputAssemblyState = &assembly_create_info;
		pipeline_create_info.pVertexInputState = &input_vertex_info;
		pipeline_create_info.pViewportState = &viewport_create_info;
		pipeline_create_info.pRasterizationState = &rasterization_create_info;
		pipeline_create_info.pMultisampleState = &multisample_create_info;
		pipeline_create_info.pDepthStencilState = &depth_stencil_create_info;
		pipeline_create_info.pColorBlendState = &color_blend_create_info;
		pipeline_create_info.pDynamicState = &dynamic_create_info;
		pipeline_create_info.layout = pipelineLayout;
		pipeline_create_info.renderPass = renderPass;
		pipeline_create_info.subpass = 0;
		pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
		vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
			&pipeline_create_info, nullptr, &pipeline);

		/***************** CLEANUP / SHUTDOWN ******************/
		// GVulkanSurface will inform us when to release any allocated resources
		shutdown.Create(vlk, [&]() {
			if (+shutdown.Find(GW::GRAPHICS::GVulkanSurface::Events::RELEASE_RESOURCES, true)) {
				CleanUp(); // unlike D3D we must be careful about destroy timing
			}
			});
	}
	void Render()
	{



		// TODO: Part 2a

		LightDir.x = -1;
		LightDir.y = 1;
		LightDir.z = 2;
		GW::MATH::GVector proxylight;
		proxylight.Create();
		proxylight.NormalizeF(LightDir, LightDir);


		LightColor.x = 0.9f;
		LightColor.y = 0.9f;
		LightColor.z = 1.0f;
		LightColor.w = 1.0f;

		// TODO: Part 4d
		proxyMan.InverseF(View, model.View);
		// grab the current Vulkan commandBuffer
		unsigned int currentBuffer;
		vlk.GetSwapchainCurrentImage(currentBuffer);
		VkCommandBuffer commandBuffer;
		vlk.GetCommandBuffer(currentBuffer, (void**)&commandBuffer);
		// what is the current client area dimensions?
		unsigned int width, height;
		win.GetClientWidth(width);
		win.GetClientHeight(height);
		// setup the pipeline's dynamic settings
		VkViewport viewport = {
			0, 0, static_cast<float>(width), static_cast<float>(height), 0, 1
		};
		VkRect2D scissor = { {0, 0}, {width, height} };
		vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
		vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

		// now we can draw
		VkDeviceSize offsets[] = { 0 };
		//vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexHandle, offsets);
		// TODO: Part 1h

	//	vkCmdBindIndexBuffer(commandBuffer, IndexHandle, 0, VK_INDEX_TYPE_UINT32);
		// TODO: Part 4d
		// TODO: Part 2i

		MeshIndex datzindex;
		unsigned int MeshyID = 0;

		for (size_t i = 0; i < OBJMESH.size(); i++)
		{
			GvkHelper::write_to_buffer(device, storageMemory[currentBuffer], &model, sizeof(SHADER_Data));
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
				pipelineLayout, 0, 1, &descriptorSet[currentBuffer], 0, nullptr);

			for (size_t j = 0; j < OBJMESH[i]->meshCount; j++)
			{

				datzindex = { OBJMESH[i]->MeshMatrix,MeshyID };
				MeshyID++;
				vkCmdBindVertexBuffers(commandBuffer, 0, 1, &OBJMESH[i]->VertexHandle, offsets);
				vkCmdBindIndexBuffer(commandBuffer, OBJMESH[i]->IndexHandle, 0, VK_INDEX_TYPE_UINT32);
				vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(MeshIndex), &datzindex);
				vkCmdDrawIndexed(commandBuffer, OBJMESH[i]->meshes[j].drawInfo.indexCount, 1, OBJMESH[i]->meshes[j].drawInfo.indexOffset, 0, 0);
			}

		}
		// TODO: Part 3b
			// TODO: Part 3d
		//TODO: part 1H
	/*	for (size_t i = 0; i < FSLogo_meshcount; i++)
		{
		vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(unsigned int), &FSLogo_meshes[i].materialIndex);
		vkCmdDrawIndexed(commandBuffer, FSLogo_meshes[i].indexCount, 1, FSLogo_meshes[i].indexOffset, 0,0);
		}*/
		//vkCmdDraw(commandBuffer, FSLogo_vertexcount, 1, 0, 0); // TODO: Part 1d, 1h


	}

	void UpdateCamera()
	{
		using namespace std::chrono;
		static auto start = std::chrono::system_clock::now();
		auto TimeTotal = std::chrono::system_clock::now();
		double timechange = std::chrono::duration<double>(TimeTotal - start).count();
		// world view and camera view holder
		GW::MATH::GMATRIXF viewHold;
		viewHold = View;


		float yaxisdown = 0.0f;
		float yaxisup = 0.0f;
		float controllerYdown = 0.0f;
		float controllerYup = 0.0f;
		const float camSpeed = 4.8f;

		playerInput.GetState(G_KEY_LEFTSHIFT, yaxisdown);
		playerInput.GetState(G_KEY_SPACE, yaxisup);
		controllerInput.GetState(0, G_LEFT_TRIGGER_AXIS, controllerYdown);
		controllerInput.GetState(0, G_RIGHT_TRIGGER_AXIS, controllerYup);

		float totalchangeinY = yaxisup - yaxisdown + controllerYup - controllerYdown;
		GW::MATH::GVECTORF camposY = { 0,(totalchangeinY * camSpeed) * static_cast<float>(timechange),0,0 };
		proxyMan.TranslateGlobalF(viewHold, camposY, viewHold);

		float frameSpeed = camSpeed * static_cast<float>(timechange);
		//wASD SETUP
		float zaxisfront = 0.0f;
		float zaxisback = 0.0f;
		float controllerzchange = 0.0f;

		playerInput.GetState(G_KEY_W, zaxisfront); //a
		playerInput.GetState(G_KEY_S, zaxisback);//d
		controllerInput.GetState(0, G_LY_AXIS, controllerzchange);

		float totalchangeinz = zaxisfront - zaxisback + controllerzchange;

		float xaxisleft = 0.0f;
		float xaxisright = 0.0f;
		float controllerxchange = 0.0f;

		playerInput.GetState(G_KEY_A, xaxisleft);//s
		playerInput.GetState(G_KEY_D, xaxisright);//w
		controllerInput.GetState(0, G_LX_AXIS, controllerxchange);

		float totalchangeinx = xaxisright - xaxisleft + controllerxchange;

		GW::MATH::GMATRIXF translationMatrix = GW::MATH::GIdentityMatrixF;
		GW::MATH::GVECTORF translationVector = { (totalchangeinx * frameSpeed),0,(totalchangeinz * frameSpeed),0 };
		proxyMan.TranslateGlobalF(translationMatrix, translationVector, translationMatrix);
		proxyMan.MultiplyMatrixF(translationMatrix, viewHold, viewHold);

		//mouse stuff

		float YaxismouseMovement = 0.0f;
		float XaxismouseMovement = 0.0f;
		float LefttoRightControls = 0.0f;
		float UpDownControls = 0.0f;

		unsigned int scHeight;
		win.GetClientHeight(scHeight);
		unsigned int scWidth;
		win.GetClientWidth(scWidth);
		float Ratio = 0.0f;
		vlk.GetAspectRatio(Ratio);
		float thumbspeed = (G2D_PI * static_cast<float>(timechange)) * 0.03;

		controllerInput.GetState(0, G_RX_AXIS, LefttoRightControls);
		controllerInput.GetState(0, G_RY_AXIS, UpDownControls);
		UpDownControls *= 0.04;
		LefttoRightControls *= 0.04;

		GW::GReturn finalresult = playerInput.GetMouseDelta(XaxismouseMovement, YaxismouseMovement);

		if ((G_PASS(finalresult) && finalresult != GW::GReturn::REDUNDANT) || UpDownControls != 0 || LefttoRightControls != 0)
		{
			float pitch = G_DEGREE_TO_RADIAN(65.0f) * (YaxismouseMovement) / scHeight + UpDownControls * (thumbspeed);
			GW::MATH::GMATRIXF pitchMatrix = GW::MATH::GIdentityMatrixF;
			proxyMan.RotateXLocalF(pitchMatrix, pitch, pitchMatrix);
			proxyMan.MultiplyMatrixF(pitchMatrix, viewHold, viewHold);

			float yaw = G_DEGREE_TO_RADIAN(65.0f) * Ratio * (XaxismouseMovement) / scWidth + LefttoRightControls * thumbspeed;
			GW::MATH::GMATRIXF yawMatrix = GW::MATH::GIdentityMatrixF;
			proxyMan.RotateYLocalF(yawMatrix, yaw, yawMatrix);
			GW::MATH::GVECTORF camLocation = viewHold.row4;
			proxyMan.MultiplyMatrixF(viewHold, yawMatrix, viewHold);
			viewHold.row4 = camLocation;
		}
		View = viewHold;
		start = std::chrono::system_clock::now();
	}
	void LoadLevel()
	{
		H2B::Parser parsy;

		parsy.Parse("barrel.h2b");


		std::string line;
		std::fstream myFile;
		myFile.open("../GameLevel.txt", std::ios::in);
		char buffer[128];
		myFile.getline(buffer, 128);

		if (myFile.is_open())
		{

			while (std::getline(myFile, line))
			{
				std::cout << line << '\n';
			}
			myFile.close();
		}


	}



private:
	void CleanUp()
	{
		// wait till everything has completed
		vkDeviceWaitIdle(device);
		// Release allocated buffers, shaders & pipeline
		// TODO: Part 1g
		vkDestroyBuffer(device, IndexHandle, nullptr);
		vkFreeMemory(device, indexData, nullptr);
		for (int i = 0; i < storageMemory.size(); ++i) {
			vkDestroyBuffer(device, storageBuffer[i], nullptr);
			vkFreeMemory(device, storageMemory[i], nullptr);
		}
		// TODO: Part 2d
		vkDestroyBuffer(device, vertexHandle, nullptr);
		vkFreeMemory(device, vertexData, nullptr);
		vkDestroyShaderModule(device, vertexShader, nullptr);
		vkDestroyShaderModule(device, pixelShader, nullptr);
		// TODO: Part 2e
		vkDestroyDescriptorSetLayout(device, descriptorlayout, nullptr);

		// TODO: part 2f
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		//vkDestroyDevice(device, nullptr);

		for (size_t i = 0; i < OBJMESH.size(); i++)
		{
			vkDestroyBuffer(device, OBJMESH[i]->VertexHandle, nullptr);
			vkFreeMemory(device, OBJMESH[i]->VertexData, nullptr);

			vkDestroyBuffer(device, OBJMESH[i]->IndexHandle, nullptr);
			vkFreeMemory(device, OBJMESH[i]->IndexData, nullptr);
		}
		unsigned lestmesh = OBJMESH.size();
		for (unsigned i = 0; i < lestmesh; ++i)
		{
			delete OBJMESH[i];
		}
	}
};
