METALCPP_DIR = ./metal-cpp
KERNEL_NAME = naive_matmul
DISPATCHER_SRC = dispatcher.cpp
BIN = run_kernel

all: compileMetal buildDispatcher run

compileMetal: $(KERNEL_NAME).metal
	xcrun -sdk macosx metal -c $(KERNEL_NAME).metal -o $(KERNEL_NAME).air
	xcrun -sdk macosx metallib $(KERNEL_NAME).air -o $(KERNEL_NAME).metallib

buildDispatcher: $(DISPATCHER_SRC)
	clang++ -std=c++17 \
	-I$(METALCPP_DIR) \
	-framework Metal -framework Foundation -framework CoreServices \
	$(DISPATCHER_SRC) -o $(BIN) 

run:
	./$(BIN)

clean:
	rm -f $(KERNEL_NAME).air $(KERNEL_NAME).metallib $(BIN)