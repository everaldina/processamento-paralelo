# processamento-paralelo
Repositotio para disciplina de processamente paralelo


# How to run
Script usado como task de build para o projeto no vscode:

```json
{
	"version": "2.0.0",
	"tasks": [
		{
			"label": "build-main",
			"type": "shell",
			"command": "C:\\msys64\\ucrt64\\bin\\g++.exe",
			"args": [
				"src/main.cpp",
				"src/image_display/save_image.cpp",
				"src/image_reader/mhd_reader.cpp", "src/metrics/ssim.cpp",
				"-I./src",
				"-O3",
				"-IC:/msys64/ucrt64/include/opencv4",
                "-LC:/msys64/ucrt64/lib",
				"-o",
				"main.exe", "-lz",
				"-lopencv_core",
                "-lopencv_imgcodecs"
			],
			"options": {
				"cwd": "${workspaceFolder}"
			},
			"group": {
				"kind": "build",
				"isDefault": true
			},
			"problemMatcher": "$gcc"
		}
	]
}
```

