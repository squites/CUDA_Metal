CUDA_PATH ?= "./examples/addOne.cu"
KERNEL ?= addOne
DISPATCHER = dispatcher.mm
OUTPUT = runner
GRID ?= 1,1,1
BLOCK ?= 256,1,1

KERNEL = $(basename $(notdir $(CUDA_PATH)))

all: $(OUTPUT)

# transpile
$(KERNEL).metal $(DISPATCHER): $(CUDA_PATH)
	python main.py $(CUDA_PATH) --grid $(GRID) --block $(BLOCK)

# compile Metal
$(KERNEL).metallib: $(KERNEL).metal
	xcrun -sdk macosx metal -c $(KERNEL).metal -o $(KERNEL).air
	xcrun -sdk macosx metallib $(KERNEL).air -o $(KERNEL).metallib

# compile dispatcher
$(OUTPUT): $(DISPATCHER) $(KERNEL).metallib
	clang++ -framework Metal -framework Foundation $(DISPATCHER) -o $(OUTPUT)

clean:
	rm -f *.air *.metallib $(OUTPUT) $(DISPATCHER)

run: $(OUTPUT)
	./$(OUTPUT)
