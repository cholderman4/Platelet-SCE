TARGET_EXEC ?= a.out

BUILD_DIR ?= ./build
SRC_DIRS ?= ./src

SRCS := $(shell find $(SRC_DIRS) -name *.cpp -or -name *.c -or -name *.s)
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CXX := g++
NVCC := nvcc

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP
CXXFLAGS=-O2 -std=c++0x -pg -g -c -Wall
NVCCFLAGS=-O2 -g -G

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(NVCC) $(OBJS) -o $@ $(LDFLAGS)


# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
    

# cuda source 
$(BUILD_DIR)/%.cu.o: %.cu
	$(MKDIR_P) $(dir $@)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) -r $(BUILD_DIR)

-include $(DEPS)

MKDIR_P ?= mkdir -p

# From: https://spin.atomicobject.com/2016/08/26/makefile-c-projects/