# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.7

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /scratch/lijingz/MasterThesis/OF_DIS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/lijingz/MasterThesis/OF_DIS/build

# Include any dependencies generated for this target.
include CMakeFiles/run_OF_RGB.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/run_OF_RGB.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/run_OF_RGB.dir/flags.make

CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o: ../run_dense.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o -c /scratch/lijingz/MasterThesis/OF_DIS/run_dense.cpp

CMakeFiles/run_OF_RGB.dir/run_dense.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_OF_RGB.dir/run_dense.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/run_dense.cpp > CMakeFiles/run_OF_RGB.dir/run_dense.cpp.i

CMakeFiles/run_OF_RGB.dir/run_dense.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_OF_RGB.dir/run_dense.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/run_dense.cpp -o CMakeFiles/run_OF_RGB.dir/run_dense.cpp.s

CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.requires

CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.provides: CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.provides

CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.provides.build: CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o


CMakeFiles/run_OF_RGB.dir/oflow.cpp.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/oflow.cpp.o: ../oflow.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/run_OF_RGB.dir/oflow.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_OF_RGB.dir/oflow.cpp.o -c /scratch/lijingz/MasterThesis/OF_DIS/oflow.cpp

CMakeFiles/run_OF_RGB.dir/oflow.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_OF_RGB.dir/oflow.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/oflow.cpp > CMakeFiles/run_OF_RGB.dir/oflow.cpp.i

CMakeFiles/run_OF_RGB.dir/oflow.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_OF_RGB.dir/oflow.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/oflow.cpp -o CMakeFiles/run_OF_RGB.dir/oflow.cpp.s

CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.requires

CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.provides: CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.provides

CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.provides.build: CMakeFiles/run_OF_RGB.dir/oflow.cpp.o


CMakeFiles/run_OF_RGB.dir/patch.cpp.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/patch.cpp.o: ../patch.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/run_OF_RGB.dir/patch.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_OF_RGB.dir/patch.cpp.o -c /scratch/lijingz/MasterThesis/OF_DIS/patch.cpp

CMakeFiles/run_OF_RGB.dir/patch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_OF_RGB.dir/patch.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/patch.cpp > CMakeFiles/run_OF_RGB.dir/patch.cpp.i

CMakeFiles/run_OF_RGB.dir/patch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_OF_RGB.dir/patch.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/patch.cpp -o CMakeFiles/run_OF_RGB.dir/patch.cpp.s

CMakeFiles/run_OF_RGB.dir/patch.cpp.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/patch.cpp.o.requires

CMakeFiles/run_OF_RGB.dir/patch.cpp.o.provides: CMakeFiles/run_OF_RGB.dir/patch.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/patch.cpp.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/patch.cpp.o.provides

CMakeFiles/run_OF_RGB.dir/patch.cpp.o.provides.build: CMakeFiles/run_OF_RGB.dir/patch.cpp.o


CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o: ../patchgrid.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o -c /scratch/lijingz/MasterThesis/OF_DIS/patchgrid.cpp

CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/patchgrid.cpp > CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.i

CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/patchgrid.cpp -o CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.s

CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.requires

CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.provides: CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.provides

CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.provides.build: CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o


CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o: ../refine_variational.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o -c /scratch/lijingz/MasterThesis/OF_DIS/refine_variational.cpp

CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/refine_variational.cpp > CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.i

CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/refine_variational.cpp -o CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.s

CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.requires

CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.provides: CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.provides

CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.provides.build: CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o


CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o: ../FDF1.0.1/image.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building C object CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o   -c /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/image.c

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/image.c > CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.i

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/image.c -o CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.s

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.requires

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.provides: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.provides

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.provides.build: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o


CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o: ../FDF1.0.1/opticalflow_aux.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building C object CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o   -c /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/opticalflow_aux.c

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/opticalflow_aux.c > CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.i

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/opticalflow_aux.c -o CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.s

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.requires

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.provides: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.provides

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.provides.build: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o


CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o: CMakeFiles/run_OF_RGB.dir/flags.make
CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o: ../FDF1.0.1/solver.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building C object CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o   -c /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/solver.c

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.i"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/solver.c > CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.i

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.s"
	/usr/bin/cc  $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /scratch/lijingz/MasterThesis/OF_DIS/FDF1.0.1/solver.c -o CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.s

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.requires:

.PHONY : CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.requires

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.provides: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.requires
	$(MAKE) -f CMakeFiles/run_OF_RGB.dir/build.make CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.provides.build
.PHONY : CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.provides

CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.provides.build: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o


# Object files for target run_OF_RGB
run_OF_RGB_OBJECTS = \
"CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o" \
"CMakeFiles/run_OF_RGB.dir/oflow.cpp.o" \
"CMakeFiles/run_OF_RGB.dir/patch.cpp.o" \
"CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o" \
"CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o" \
"CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o" \
"CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o" \
"CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o"

# External object files for target run_OF_RGB
run_OF_RGB_EXTERNAL_OBJECTS =

run_OF_RGB: CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/oflow.cpp.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/patch.cpp.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/build.make
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_xphoto.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_xobjdetect.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_tracking.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_surface_matching.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_structured_light.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_stereo.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_saliency.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_rgbd.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_reg.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_plot.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_optflow.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_line_descriptor.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_fuzzy.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_dpm.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_dnn.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_datasets.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_ccalib.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_bioinspired.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_bgsegm.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_aruco.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_videostab.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_superres.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_stitching.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_photo.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_text.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_face.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_ximgproc.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_xfeatures2d.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_shape.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_video.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_objdetect.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_calib3d.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_features2d.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_ml.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_highgui.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_videoio.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_imgcodecs.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_imgproc.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_flann.so.3.1.0
run_OF_RGB: /scratch_net/unclemax/lijingz/anaconda2/lib/libopencv_core.so.3.1.0
run_OF_RGB: CMakeFiles/run_OF_RGB.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Linking CXX executable run_OF_RGB"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/run_OF_RGB.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/run_OF_RGB.dir/build: run_OF_RGB

.PHONY : CMakeFiles/run_OF_RGB.dir/build

CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/run_dense.cpp.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/oflow.cpp.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/patch.cpp.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/patchgrid.cpp.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/refine_variational.cpp.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/image.c.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/opticalflow_aux.c.o.requires
CMakeFiles/run_OF_RGB.dir/requires: CMakeFiles/run_OF_RGB.dir/FDF1.0.1/solver.c.o.requires

.PHONY : CMakeFiles/run_OF_RGB.dir/requires

CMakeFiles/run_OF_RGB.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/run_OF_RGB.dir/cmake_clean.cmake
.PHONY : CMakeFiles/run_OF_RGB.dir/clean

CMakeFiles/run_OF_RGB.dir/depend:
	cd /scratch/lijingz/MasterThesis/OF_DIS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/lijingz/MasterThesis/OF_DIS /scratch/lijingz/MasterThesis/OF_DIS /scratch/lijingz/MasterThesis/OF_DIS/build /scratch/lijingz/MasterThesis/OF_DIS/build /scratch/lijingz/MasterThesis/OF_DIS/build/CMakeFiles/run_OF_RGB.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/run_OF_RGB.dir/depend

