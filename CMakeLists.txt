add_definitions("-DFLANN_STATIC -DDISABLE_OPENNI -DAUDIOIMPL_IS_REMOTE -DBOOST_ASIO_DISABLE_STD_CHRONO -DBOOST_FILESYSTEM_VERSION=3")
include_directories(. 
	lib/configs/
	lib/pcl/common/include/
	lib/pcl/features/include/
	lib/pcl/search/include/
	lib/pcl/kdtree/include/
	lib/pcl/2d/include/
	lib/pcl/octree/include/
	lib/pcl/sample_consensus/include/
	lib/pcl/segmentation/include/
	lib/pcl/recognition/include/
	lib/pcl/filters/include/
	lib/pcl/ml/include/
	lib/pcl/registration/include/
	lib/pcl/geometry/include/
	lib/pcl/io/include/
	lib/eigen/
	lib/flann/src/cpp/
	)

file(GLOB_RECURSE SERVICES_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "services/*.cpp")
file(GLOB_RECURSE TESTS_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "tests/*.cpp")

list(APPEND FLANN_CPP "lib/flann/src/cpp/flann/flann.cpp" "lib/flann/src/cpp/flann/flann_cpp.cpp")
file(GLOB_RECURSE FLANN_C RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/flann/src/*.c")
file(GLOB_RECURSE PCL_COMMON_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/common/src/*.cpp")
file(GLOB_RECURSE PCL_COMMON_C RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/common/src/*.c")
file(GLOB_RECURSE PCL_FEATURES_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/features/src/*.cpp")
file(GLOB_RECURSE PCL_FILTERS_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/filters/src/*.cpp")
file(GLOB_RECURSE PCL_KDTREE_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/kdtree/src/*.cpp")
file(GLOB_RECURSE PCL_ML_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/ml/src/*.cpp")
list(APPEND PCL_OCTREE_CPP "lib/pcl/octree/src/octree_inst.cpp")
file(GLOB_RECURSE PCL_RECOGNITION_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/recognition/src/*.cpp")
list(REMOVE_ITEM PCL_RECOGNITION_CPP
	"lib/pcl/recognition/src/face_detection/face_detector_data_provider.cpp" 
	"lib/pcl/recognition/src/face_detection/rf_face_detector_trainer.cpp")
file(GLOB_RECURSE PCL_REGISTRATION_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/registration/src/*.cpp")
list(REMOVE_ITEM PCL_REGISTRATION_CPP
	"lib/pcl/registration/src/pairwise_graph_registration.cpp")
file(GLOB_RECURSE PCL_SAMPLE_CONSENSUS_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/sample_consensus/src/*.cpp")
file(GLOB_RECURSE PCL_SEARCH_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/search/src/*.cpp")
file(GLOB_RECURSE PCL_SEGMENTATION_CPP RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "lib/pcl/segmentation/src/*.cpp")
list(APPEND PCL_IO_CPP "lib/pcl/io/src/pcd_io.cpp"
	"lib/pcl/io/src/lzf.cpp"
)

qi_create_lib(pcl_plugin SHARED 
	${SERVICES_CPP} 
 	${TESTS_CPP}	
	${FLANN_CPP} 
	${FLANN_C} 
	${PCL_COMMON_CPP}
	${PCL_COMMON_C}
	${PCL_FEATURES_CPP}
	${PCL_FILTERS_CPP}
	${PCL_KDTREE_CPP}
	${PCL_ML_CPP}
	${PCL_OCTREE_CPP}
	${PCL_RECOGNITION_CPP}	
	${PCL_REGISTRATION_CPP}
	${PCL_SAMPLE_CONSENSUS_CPP}
	${PCL_SEARCH_CPP}
	${PCL_SEGMENTATION_CPP}
	${PCL_IO_CPP}	
)
qi_use_lib(pcl_plugin self utils tinythread++)
qi_stage_lib(pcl_plugin)

