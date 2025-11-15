CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall

TARGET = build/chess_engine
SRC = chess_engine.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)
	@echo "Build complete! Engine binary: $(TARGET)"

clean:
	rm -rf build

.PHONY: all clean

