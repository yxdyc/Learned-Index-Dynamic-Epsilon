cmake_minimum_required(VERSION 3.12)
project(LearnedIndexOnlineLearning)

set(CMAKE_CXX_STANDARD 14)

add_subdirectory(/home/xxx/dlib-19.19/dlib dlib_build )
add_subdirectory(/home/xxx/work/online_index/lib/cnpy_build/cnpy-master)
include_directories(include lib rmi_models)

#find_package(gflags REQUIRED)

add_subdirectory(lib/gflags)

aux_source_directory(include model_list)


IF(CMAKE_BUILD_TYPE STREQUAL Debug)
    ADD_DEFINITIONS(-DDebug)
ENDIF()

file(GLOB RADIX_SPLINE "include/radix_spline/*.h")
file(GLOB MET "METIndexer.h")


# Set this for dataset where key type is float, e.g., the MAP dataset
# ADD_DEFINITIONS(-DFLOAT_KEY)

set(COMMON_SOURCES include/DataProcessor.hpp include/PgmIndexer.hpp include/PgmIndexer_Modified.hpp include/IndexMechanism.hpp
        include/BtreeIndexer.hpp include/RegressionIndexer.hpp include/FittingTreeIndexer.hpp include/FittingTreeIndexer_Modified.hpp
        include/RMIIndexerLinear.hpp include/cxxopts.hpp lib/cnpy.h include/epsilon_meta_learner.hpp)

#add_executable(DifferentDt different_dts.cpp ${COMMON_SOURCES})
#add_executable(VariedEpsilon varied_epsilon.cpp ${COMMON_SOURCES})
add_executable(DynamicEpsilon_FLOAT index_dynamic_epsilon.cpp ${COMMON_SOURCES} ${RADIX_SPLINE} ${MET})
add_executable(DynamicEpsilon_INT index_dynamic_epsilon.cpp ${COMMON_SOURCES} ${RADIX_SPLINE} ${MET})

target_compile_definitions(DynamicEpsilon_FLOAT PUBLIC -DFLOAT_KEY)

target_link_libraries(DynamicEpsilon_INT dlib::dlib)
target_link_libraries(DynamicEpsilon_FLOAT dlib::dlib -DFLOAT_KEY)


