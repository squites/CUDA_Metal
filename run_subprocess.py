import subprocess

KERNEL_NAME = "vecAdd"
METALCPP_PATH = "-I./metal-cpp"

#app = subprocess.run(["python", "main.py"], capture_output=True, text=True)
#print(app.stdout)

print("Compiling metallib...")
xcrun = "/usr/bin/xcrun"
clang = "/usr/bin/clang"
subprocess.run([xcrun, "-sdk", "macosx", "metal", "-c", KERNEL_NAME+".metal", "-o", KERNEL_NAME+".air"])
subprocess.run([xcrun, "-sdk", "macosx", "metallib", KERNEL_NAME+".air", "-o", KERNEL_NAME+"metallib"])

print("Compiling kernel...")
subprocess.run([clang, METALCPP_PATH, "-framework", "Metal", "-framework", "Foundation", "-framework", "CoreServices", "dispatcher.cpp", "-o", KERNEL_NAME])

print("Executing kernel...")
res = subprocess.run(["./"+KERNEL_NAME])
print(res.stdout)

#clang++ -std=c++17 -I./metal-cpp -framework Metal -framework Foundation -framework CoreServices dispatcher.cpp -o vecAdd 