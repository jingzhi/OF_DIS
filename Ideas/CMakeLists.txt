cmake_minimum_required(VERSION 2.8)
project(Idea)
find_package(OpenCV REQUIRED)
find_package(PythonLibs 2.7)

add_executable(benchmark benchmark.cpp )
target_include_directories(benchmark PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(benchmark ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

add_executable(occlu occlu.cpp )
target_include_directories(occlu PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(occlu ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

add_executable(grad grad.cpp )
target_include_directories(grad PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(grad ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

add_executable(errormap errormap.cpp )
target_include_directories(errormap PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(errormap ${OpenCV_LIBS} ${PYTHON_LIBRARIES})

add_executable(evalall evalall.cpp )
target_include_directories(evalall PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(evalall ${OpenCV_LIBS} ${PYTHON_LIBRARIES})
