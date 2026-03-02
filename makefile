KERNEL ?= addOne
DISPATCHER = dispatcher.mm
OUTPUT = runner

all: $(OUTPUT)

# transpile
$(KERNEL).metal $(DISPATCHER): $(KERNEL).cu
	python main.py $(KERNEL).cu --grid $(GRID) --block $(BLOCK)

# compile Metal
$(KERNEL).metallib: $(KERNEL).metal
	xcrun -sdk macosx metal -c $(KERNEL).metal -o $(KERNEL).air
	xcrun -sdk macosx metallib $(KERNEL).air -o $(KERNEL).metallib

# compile dispatcher
$(OUTPUT): $(DISPATCHER) $(KERNEL).metallib
	clang++ -framework Metal -framework Foundation $(DISPATCHER) -o $(OUTPUT)

run: $(OUTPUT)
	./$(OUTPUT)

clean:
	rm -f *.air *.metallib $(OUTPUT)