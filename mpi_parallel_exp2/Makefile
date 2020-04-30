CXX := mpic++
CFLAGS := -Wall -g
TARGET := neuralnet

# $(wildcard *.cpp /xxx/xxx/*.cpp): get all .cpp files from the current directory and dir "/xxx/xxx/"
SRCS := $(wildcard *.cpp)
# $(patsubst %.cpp,%.o,$(SRCS)): substitute all ".cpp" file name strings to ".o" file name strings
OBJS := $(patsubst %.cpp,%.o,$(SRCS))

all: $(TARGET)
$(TARGET): $(OBJS)
	$(CXX) -g -o $@ $^
objects/%.o: %.cpp
	$(CXX) $(CFLAGS) -g -c $<
clean:
	rm -rf $(TARGET) *.o
	
.PHONY: all clean