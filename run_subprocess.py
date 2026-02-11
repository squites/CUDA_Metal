import subprocess

KERNEL_NAME = "naive_matmul"
METALCPP_PATH = "-I./metal-cpp"

print("Generating Metal Shader...")
transpiler = subprocess.run(["python", "main.py"], capture_output=True, text=True)
print(transpiler.stdout)

print("Compiling and executing kernel...")
subprocess.run(["make", "all"], shell=True)
