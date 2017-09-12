/**
* Copyright 2017 IBM Corp. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*
*/

//#define VISUALIZATION           1
//#define WRITE_SCENE_PCD         1

#include "PCLObjectRecognition.h"
#include "SelfInstance.h"

#include "opencv2/opencv.hpp"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/registration/icp.h>

#ifdef VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#endif

REG_SERIALIZABLE(PCLObjectRecognition);
REG_OVERRIDE_SERIALIZABLE(IObjectRecognition, PCLObjectRecognition);
RTTI_IMPL(PCLObjectRecognition, IObjectRecognition);

PCLObjectRecognition::PCLObjectRecognition() :
    IObjectRecognition( "PCL", AUTH_NONE ),
    m_ClusterSizeMin1( 0.01f ),
    m_ClusterSizeMax1( 1.0f ),
    m_ClusterSizeMin2( 0.1f ),
    m_ClusterSizeMax2( 1.0f ),
    m_ClusterTolerance1( 0.01f ),
    m_ClusterTolerance2( 0.2f ),
    m_AcceptableFittingScore( 0.00075f ),
    m_DivRange( 0.125f ),
    m_MeanK( 0.0f ),
    m_SamplingSize( 0.005f ),
    m_HistoryTerm( 10 )
{}

void PCLObjectRecognition::Serialize(Json::Value& json)
{
    IObjectRecognition::Serialize(json);
    json["m_ClusterSizeMin1"] = m_ClusterSizeMin1;
    json["m_ClusterSizeMax1"] = m_ClusterSizeMax1;
    json["m_ClusterSizeMin2"] = m_ClusterSizeMin2;
    json["m_ClusterSizeMax2"] = m_ClusterSizeMax2;
    json["m_ClusterTolerance1"] = m_ClusterTolerance1;
    json["m_ClusterTolerance2"] = m_ClusterTolerance2;
    json["m_AcceptableFittingScore"] = m_AcceptableFittingScore;
    json["m_DivRange"] = m_DivRange;
    json["m_MeanK"] = m_MeanK;
    json["m_SamplingSize"] = m_SamplingSize;
    json["m_HistoryTerm"] = m_HistoryTerm;

    SerializeVector( "m_Objects", m_Objects, json );
}

void PCLObjectRecognition::Deserialize(const Json::Value& json)
{
    IObjectRecognition::Deserialize(json);

    if ( json["m_ClusterSizeMin1"].isNumeric() )
    {
        m_ClusterSizeMin1 = json["m_ClusterSizeMin1"].asFloat();
    }

    if ( json["m_ClusterSizeMax1"].isNumeric() )
    {
        m_ClusterSizeMax1 = json["m_ClusterSizeMax1"].asFloat();
    }

    if ( json["m_ClusterSizeMin2"].isNumeric() )
    {
        m_ClusterSizeMin2 = json["m_ClusterSizeMin2"].asFloat();
    }

    if ( json["m_ClusterSizeMax2"].isNumeric() )
    {
        m_ClusterSizeMax2 = json["m_ClusterSizeMax2"].asFloat();
    }

    if ( json["m_ClusterTolerance1"].isNumeric() )
    {
        m_ClusterTolerance1 = json["m_ClusterTolerance1"].asFloat();
    }

    if ( json["m_ClusterTolerance2"].isNumeric() )
    {
        m_ClusterTolerance2 = json["m_ClusterTolerance2"].asFloat();
    }

    if ( json["m_AcceptableFittingScore"].isNumeric() )
    {
        m_AcceptableFittingScore = json["m_AcceptableFittingScore"].asFloat();
    }

    if ( json["m_DivRange"].isNumeric() )
    {
        m_DivRange = json["m_DivRange"].asFloat();
    }

    if ( json["m_MeanK"].isNumeric() )
    {
        m_MeanK = json["m_MeanK"].asFloat();
    }

    if ( json["m_SamplingSize"].isNumeric() )
    {
        m_SamplingSize = json["m_SamplingSize"].asFloat();
    }

    if ( json["m_HistoryTerm"].isNumeric() )
    {
        m_HistoryTerm = json["m_HistoryTerm"].asFloat();
    }

    DeserializeVector( "m_Objects", json, m_Objects );

    // if no data is provided, initialize with some default data..
    if ( m_Objects.size() == 0 )
    {
        std::vector<std::string> models;
        models.push_back( "shared/pcd/nasa_drill/nasa_drill.pcd" );
        m_Objects.push_back( ObjectModel( "nasa_drill", models ) );
    }
}

bool PCLObjectRecognition::Start()
{
    if (! IObjectRecognition::Start() )
    {
        return false;
    }

    int modelsLoaded = 0;

    for (size_t i = 0; i < m_Objects.size(); ++i)
    {
        if ( m_Objects[i].LoadPCD() )
        {
            Log::Status( "PCLObjectRecognition", "Loaded models for %s", m_Objects[i].m_ObjectId.c_str() );
            modelsLoaded += 1;
        }
        else
        {
            Log::Error( "PCLObjectRecognition", "Failed to load models for object %s", m_Objects[i].m_ObjectId.c_str() );
        }

        m_Objects[i].setHistoryTerm( m_HistoryTerm );
    }

    Log::Status( "PCLObjectRecognition", "Loaded %d models", modelsLoaded );

#ifdef VISUALIZATION
    viewer = new pcl::visualization::PCLVisualizer("Object Localiser");
    viewer->setSize(1600, 900);
    viewer->setShowFPS(false);
    viewer->setCameraPosition(0.0f, 0.0f, -1.0f, 0.0f, -1.0f, 0.0f);
#endif

    return true;
}

void PCLObjectRecognition::ClassifyObjects(const std::string& a_DepthImageData,
        OnClassifyObjects a_Callback )
{
    ThreadPool::Instance()->InvokeOnThread<ProcessDepthData*>( DELEGATE( PCLObjectRecognition, ProcessThread, ProcessDepthData*, this ),
            new ProcessDepthData( a_DepthImageData, a_Callback ) );
}

void PCLObjectRecognition::ProcessThread( ProcessDepthData* a_pData )
{
    double startTime = Time().GetEpochTime();
//  Log::Status( "PCLObjectRecognition", "Processing %u bytes of depth data.", a_pData->m_DepthData.size() );

#if 0
    FILE* fp = fopen( "scene.png", "wb" );

    if ( fp != NULL )
    {
        fwrite( a_pData->m_DepthData.data(), 1, a_pData->m_DepthData.size(), fp );
        fclose( fp );
    }

#endif

    // convert depth data into PCD
    const std::string& data = a_pData->m_DepthData;
    std::vector<unsigned char> encoded( (unsigned char*)data.data(), (unsigned char*)data.data() + data.size() );
    cv::Mat decoded = cv::imdecode( encoded, CV_LOAD_IMAGE_ANYDEPTH );

    pcl::PointCloud<pcl::PointXYZ>::Ptr spScene( new pcl::PointCloud<PointType>() );
    spScene->height = decoded.rows;
    spScene->width = decoded.cols;
    spScene->is_dense = false;
    spScene->points.resize( spScene->width * spScene->height );

    const float constant = 1.0f / 570;
    const int centerX = spScene->width >> 1;
    const int centerY = spScene->height >> 1;
    register int depth_idx = 0;

    for (int v = -centerY; v < centerY; ++v)
    {
        for (register int u = -centerX; u < centerX; ++u, ++depth_idx)
        {
            pcl::PointXYZ& pt = spScene->points[depth_idx];
            pt.z = decoded.at<unsigned short>( depth_idx ) * 0.001f;
            pt.x = static_cast<float>(u) * pt.z * constant;
            pt.y = static_cast<float>(v) * pt.z * constant;
        }
    }

    spScene->sensor_origin_.setZero();
    spScene->sensor_orientation_.w() = 0.0f;
    spScene->sensor_orientation_.x() = 1.0f;
    spScene->sensor_orientation_.y() = 0.0f;
    spScene->sensor_orientation_.z() = 0.0f;

    ObjectModel& o = m_Objects[0];

    pcl::PointCloud<PointType>::Ptr scene(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*spScene, *scene);

#if WRITE_SCENE_PCD
    // save to a local file
    pcl::io::savePCDFile( "scene.pcd", *scene );
#endif

    //
    // Down sampling
    //
    pcl::PointCloud<PointType> scene_sample; // reduced point cloud
    pcl::VoxelGrid<PointType> us;
    us.setInputCloud(scene);
    us.setLeafSize(m_SamplingSize, m_SamplingSize, m_SamplingSize);
    us.filter(scene_sample);
    // Copy reduced point cloud to target point cloud 
    scene->points.swap(scene_sample.points);
    scene->width = scene_sample.width;
    scene->height = scene_sample.height;

#ifdef VISUALIZATION
    // Keep original scene for visualization
    pcl::PointCloud<PointType>::Ptr scene_all(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*scene, *scene_all);
#endif

    //
    // Exclude point further than 1.5 meter
    //
    pcl::PointCloud<PointType> scene_clip; // clipped point cloud
    pcl::PassThrough<PointType> ptf;
    ptf.setInputCloud(scene);
    // Set clipping z range within 1.5 meter
    ptf.setFilterFieldName("z");
    ptf.setFilterLimits(0.0, 1.5);
    ptf.filter(scene_clip);
    // Copy clipped point cloud to target point cloud
    scene->points.swap(scene_clip.points);
    scene->width = scene_clip.width;
    scene->height = scene_clip.height;

    // Check if enough points exists in this z range
    if (scene->points.size() < o.m_PCD[0].m_Model->points.size() / 2) {
    //  Log::Status( "PCLObjectRecognition", "No object in 1 meter range");
        a_pData->m_Results["objects"] = Json::Value( Json::arrayValue );
        Json::Value result;
        result["objectId"] = -1;
        result["confidence"] = 0.0;
        Json::Value tf;
        tf["x"] = 0.0;
        tf["y"] = 0.0;
        tf["z"] = 0.0;
        result["translation"] = tf;
        Json::Value rot;
        rot["x"] = 0.0;
        rot["y"] = 0.0;
        rot["z"] = 0.0;
        rot["w"] = 1.0;
        result["rotation"] = rot;
        a_pData->m_Results["objects"].append(result);
#ifdef VISUALIZATION
        // Visualize only non-target point cloud
        viewer->removeAllPointClouds();
        pcl::visualization::PointCloudColorHandlerCustom<PointType> all_color_handler(scene_all, 128, 128, 128);
        viewer->addPointCloud(scene_all, all_color_handler, "scene_all");
        viewer->spinOnce();
#endif // VISUALIZATION
        return;
    }

    //
    // Smooth surface by removing noise
    //
    if (m_MeanK > 0.0) {
        pcl::PointCloud<PointType> scene_revise; // revised point cloud
        pcl::StatisticalOutlierRemoval<PointType> sor;
        sor.setInputCloud(scene);
        sor.setMeanK(m_MeanK); // set smoothing level
        sor.setStddevMulThresh(1.0);
        sor.filter(scene_revise);
        // Copy revised point cloud to target point cloud
        scene->points.swap(scene_revise.points);
        scene->width = scene_revise.width;
        scene->height = scene_revise.height;
    }

#ifdef VISUALIZATION
    // Keep original scene for visualization
    pcl::PointCloud<PointType>::Ptr scene_1m(new pcl::PointCloud<PointType>());
    pcl::copyPointCloud(*scene, *scene_1m);
#endif

#if WRITE_SCENE_PCD
    // save to a local file
    pcl::io::savePCDFile( "scene_1M.pcd", *scene );
#endif

    //
    // Extract clusters which have equal or less points than the reference model
    //
    int nclusters = 0; // number of clusters
    std::vector<pcl::PointCloud<PointType>::Ptr> cluster; // clustered point clouds

    pcl::search::KdTree<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    tree->setInputCloud(scene);
    std::vector<pcl::PointIndices> cindices; // index arrays for cluster separation
    pcl::EuclideanClusterExtraction<PointType> ec;
    // Set tolerance, minimum size, and maximum size for clustering stage 1
    ec.setClusterTolerance(m_ClusterTolerance1);
    ec.setMinClusterSize((int)((float)o.m_PCD[0].m_Model->points.size() * m_ClusterSizeMin1));
    ec.setMaxClusterSize((int)((float)o.m_PCD[0].m_Model->points.size() * m_ClusterSizeMax1));
    ec.setSearchMethod(tree);
    ec.setInputCloud(scene);
    // Extract clustering result indices
    ec.extract(cindices);

    //
    // Divide scene to each cluster
    //
    int cn = cindices.end() - cindices.begin() + 1; // number of clusters in result
    cluster.resize(cn);
    // loop for each cluster
    for (std::vector<pcl::PointIndices>::const_iterator it = cindices.begin(); it != cindices.end(); it++) {
        // Allocate point cloud object for a cluster
        cluster[nclusters] = (pcl::PointCloud<PointType>::Ptr)(new pcl::PointCloud<PointType>());
        // Loop for each point
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
            // Copy point coordinates from target point cloud to clustered point cloud
            cluster[nclusters]->points.push_back(scene->points[*pit]);
        // Set basic information for this clustered point cloud
        cluster[nclusters]->width = cluster[nclusters]->points.size();
        cluster[nclusters]->height = 1;
        cluster[nclusters]->is_dense = true;
        nclusters++;
    }

    //
    // Merge the extracted clusters to single scene
    //
    pcl::PointCloud<PointType>::Ptr merged(new pcl::PointCloud<PointType>()); // Merged point cloud
    // Loop for each clustered point cloud
    for (int i = 0; i < nclusters; i++) {
        *merged += *cluster[i];
    }
    // Copy merged point cloud to target point cloud
    scene->points.swap(merged->points);
    scene->width = merged->width;
    scene->height = merged->height;

#if WRITE_SCENE_PCD
    // save to a local file
    pcl::io::savePCDFile( "scene_stage1.pcd", *scene );
#endif

    //
    // Extract larger clusters which have similar number of points
    //
    nclusters = 0;
    std::vector<pcl::PointCloud<PointType>::Ptr> cluster2; // clustered point clouds for stage 2
    std::vector<Eigen::Vector4f> centroid2; // center position for each cluster
    std::vector<float> score; // ICP fitting score (lower score is better)
    std::vector<float> div; // divergence from moving average of controid position
    std::vector<Eigen::Matrix4f> pose; // pose of the model which is ICPed on each cluster

    pcl::search::KdTree<PointType>::Ptr tree2(new pcl::search::KdTree<PointType>);
    tree2->setInputCloud(scene);
    std::vector<pcl::PointIndices> cindices2; // index arrays for cluster separation
    pcl::EuclideanClusterExtraction<PointType> ec2;
    // Set tolerance, minimum size, and maximum size for clustering stage 2
    ec2.setClusterTolerance(m_ClusterTolerance2);
    ec2.setMinClusterSize((int)((float)o.m_PCD[0].m_Model->points.size() * m_ClusterSizeMin2));
    ec2.setMaxClusterSize((int)((float)o.m_PCD[0].m_Model->points.size() * m_ClusterSizeMax2));
    ec2.setSearchMethod(tree2);
    ec2.setInputCloud(scene);
    // Extract clustering result indices
    ec2.extract(cindices2);

    //
    // Divide scene to the clusters
    //
    cn = cindices2.end() - cindices2.begin() + 1; // number of clusters in result
    // Resize array size to number of clusters
    cluster2.resize(cn);
    centroid2.resize(cn);
    score.resize(cn);
    div.resize(cn);
    pose.resize(cn);
    // loop for each cluster
    for (std::vector<pcl::PointIndices>::const_iterator it = cindices2.begin(); it != cindices2.end(); it++) {
        // Allocate point cloud object for a cluster
        cluster2[nclusters] = (pcl::PointCloud<PointType>::Ptr)(new pcl::PointCloud<PointType>());
        // Loop for each point
        for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); pit++)
            // Copy point coordinates from target point cloud to clustered point cloud
            cluster2[nclusters]->points.push_back(scene->points[*pit]);
        // Set basic information for this clustered point cloud
        cluster2[nclusters]->width = cluster2[nclusters]->points.size();
        cluster2[nclusters]->height = 1;
        cluster2[nclusters]->is_dense = true;

        //
        // Apply ICP between each cluster and the reference model
        //
        pcl::compute3DCentroid(*cluster2[nclusters], centroid2[nclusters]);
        Eigen::Matrix4f p; // transformation matrix to align reference model on each cluster
        p <<
            1.0, 0.0, 0.0, centroid2[nclusters][0],
            0.0, 1.0, 0.0, centroid2[nclusters][1],
            0.0, 0.0, 1.0, centroid2[nclusters][2],
            0.0, 0.0, 0.0, 1.0;
        pcl::PointCloud<PointType>::Ptr model_on_cluster(new pcl::PointCloud<PointType>()); // aligned reference model
        // move reference model to centroid of a cluster
        pcl::transformPointCloud(*o.m_PCD[0].m_Model, *model_on_cluster, p);
        // Try to apply ICP to check fitting score
        pcl::IterativeClosestPoint<PointType, PointType> icpc;
        icpc.setInputSource(model_on_cluster);
        icpc.setInputTarget(cluster2[nclusters]);
        pcl::PointCloud<PointType> icpc_result; // ICPed reference model point cloud
        // Calculate closest pose of reference model on a cluster
        icpc.align(icpc_result);
        // Check if converged
        if (icpc.hasConverged()) {
            // Calculate fitting score (lower score is better)
            score[nclusters] = icpc.getFitnessScore();
            // Get transformation matrix to move reference model to closest pose
            Eigen::Matrix4f adj = icpc.getFinalTransformation();
            // Calculate absolute transformation from zero point
            pose[nclusters] = adj * p;
            Eigen::Vector3f t = pose[nclusters].block<3, 1>(0, 3);

            // Get divergence from moving avegare position
            div[nclusters] = o.getMovingAveDivergence(t);
        }
        else {
            // Set worst score
            score[nclusters] = 1.0;
            // Set worst divergence
            div[nclusters] = m_DivRange;
        }
        nclusters++;
    }

    // Find highest fitting cluster
    float best = 1.0f;
    int ibest = -1;
    for (int i = 0; i < nclusters; i++) {
        if (score[i] < best) {
            best = score[i];
            ibest = i;
        }
    }

#if WRITE_SCENE_PCD
    // save to a local file
    if (ibest >= 0)
    pcl::io::savePCDFile( "scene_stage2.pcd", *cluster2[ibest] );
#endif

    int found; // find decision if object localiser find the target object or not
    Eigen::Vector3f translation; // final translation from zero point
    Eigen::Matrix3f rotation;

    if (ibest < 0 || score[ibest] > m_AcceptableFittingScore) {
        // Fitting score too bad
    //  Log::Status( "PCLObjectRecognition", "No candidates found");
        found = false;
    }
    else {
        // set translation of the best fit cluster as the final translation
        translation = pose[ibest].block<3, 1>(0, 3);
        // store translation as the latest position of moving average
        o.updateMovingAve(translation);

        // Check if the distane from moving average is in the acceptance range
        if (div[ibest] >= m_DivRange) {
            // Too far from moving average position
        //  Log::Status( "PCLObjectRecognition", "Candidate rejected\n        => Divergence (%1.3f) is too far from moving average", div[ibest]);
            found = false;
        }
        else {
            // Accept the result
            rotation = pose[ibest].block<3, 3>(0, 0);
            found = true;
        }
    }

    // find objects in PCD
    a_pData->m_Results["objects"] = Json::Value( Json::arrayValue );

    Json::Value result;

    if (found) {
        Eigen::Quaternionf q(rotation);
        Log::Status( "PCLObjectRecognition", "Object detected\n        t = (%1.3f, %1.3f, %1.3f), r = (%1.3f, %1.3f, %1.3f, %1.3f)",
            translation(0), translation(1), translation(2), -q.x(), q.y(), q.z(), q.w());

        result["confidence"] = score[ibest] < m_AcceptableFittingScore / 2.0 ?
                1.0: m_AcceptableFittingScore / 2.0 / score[ibest];

        result["objectId"] = o.m_ObjectId;
        //
        // x <- (z), y <- (-x), z <- (-y)
        //
        Json::Value tf;
        tf["x"] = translation(2);
        tf["y"] = -translation(0);
        tf["z"] = -translation(1);
        result["translation"] = tf;

        Json::Value rot;
        //
        // x <- (y), y <- (x), z <- (-z), w <- (w)
        //
        rot["x"] = q.y();
        rot["y"] = q.x();
        rot["z"] = -q.z();
        rot["w"] = q.w();
        result["rotation"] = rot;

    }
    else {
        result["objectId"] = -1;
        result["confidence"] = 0.0;
        Json::Value tf;
        tf["x"] = 0.0;
        tf["y"] = 0.0;
        tf["z"] = 0.0;
        result["translation"] = tf;

        Json::Value rot;
        rot["x"] = 0.0;
        rot["y"] = 0.0;
        rot["z"] = 0.0;
        rot["w"] = 1.0;
        result["rotation"] = rot;
    }

    a_pData->m_Results["objects"].append(result);

#ifdef VISUALIZATION
  viewer->removeAllPointClouds();

  // Show whole scene on blue
  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_all_color_handler(scene_all, 128, 128, 128);
  viewer->addPointCloud(scene_all, scene_all_color_handler, "scene_all");

  // Show scene in 1M range on green
  pcl::visualization::PointCloudColorHandlerCustom<PointType> scene_color_handler(scene_1m, 255, 255, 255);
  viewer->addPointCloud(scene_1m, scene_color_handler, "scene_1m");

  // Show candidate clusters on red
  for (int i = 0; i < nclusters; i++) {
    int c = (i % 5) + 1;
    int r = c & 4 ? 255: 192;
    int g = c & 2 ? 255: 192;
    int b = c & 1 ? 255: 192;
    if (i == ibest) {
      // Show detected object on yellow (acceptable) or on pink (unacceptable)
      if (found) {
        r = 255;
        g = 255;
        b = 0;
      }
      else {
        r = 255;
        g = 0;
        b = 0;
      }
    }
    pcl::visualization::PointCloudColorHandlerCustom<PointType> cluster_color_handler(cluster2[i], r, g, b);
    std::stringstream ss;
    ss << "cluster" << i;
    viewer->addPointCloud(cluster2[i], cluster_color_handler, ss.str());
  }

  for (int i = 0; i < 100; i++)
    viewer->spinOnce();

#endif // VISUALIZATION

    double elapsed = Time().GetEpochTime() - startTime;
//  Log::Status( "PCLObjectRecognition", "Recognition completed in %.2f seconds, found %u objects", elapsed, a_pData->m_Results["objects"].size() );

    // return results..
    ThreadPool::Instance()->InvokeOnMain<ProcessDepthData*>( DELEGATE( PCLObjectRecognition, SendResults, ProcessDepthData*, this ), a_pData );
}

void PCLObjectRecognition::SendResults( ProcessDepthData* a_pData )
{
    a_pData->m_Callback( a_pData->m_Results );
    delete a_pData;
}


bool PCLObjectRecognition::ObjectModel::LoadPCD()
{
    const std::string& staticData = Config::Instance()->GetStaticDataPath();

    for (size_t i = 0; i < m_Models.size(); ++i)
    {
        std::string modelPath( staticData + m_Models[i] );

        ModelPCD pcd;

        if ( pcl::io::loadPCDFile( modelPath, *pcd.m_Model ) < 0 )
        {
            Log::Error( "ObjectModel", "Failed to load %s", modelPath.c_str() );
            return false;
        }

        m_PCD.push_back( pcd );
    }

    return true;
}

/*
 * Calculate moving average from past position of the target object
 */
float PCLObjectRecognition::ObjectModel::getMovingAveDivergence(Eigen::Vector3f t)
{
    float d = 0.0f;

    // Check if there is history
    if (m_nHistory > 0) {
        // Calculate moving average
        Eigen::Vector3f tma; // moving average of 3D position
        tma[0] = tma[1] = tma[2] = 0.0f;
        // Loop for number od histories
        for (int i = 0; i < m_nHistory; i++) {
            // Add x, y, and z coordinates
            tma[0] += m_TranslationHistory[i][0];
            tma[1] += m_TranslationHistory[i][1];
            tma[2] += m_TranslationHistory[i][2];
        }
        // Calculate average position for x, y, and z coordinates
        tma[0] /= (float)m_nHistory;
        tma[1] /= (float)m_nHistory;
        tma[2] /= (float)m_nHistory;

        // Calculate divergence (distance) from moving average
        d = sqrt((t[0] - tma[0]) * (t[0] - tma[0])
            + (t[1] - tma[1]) * (t[1] - tma[1])
            + (t[2] - tma[2]) * (t[2] - tma[2]));
    }

    return d;
}

/*
 * Update moving average with the latest position
 */
void PCLObjectRecognition::ObjectModel::updateMovingAve(Eigen::Vector3f t)
{
    // Shift moving average history
    for (int i = m_HistoryTerm - 2; i >= 0; i--) {
        m_TranslationHistory[i + 1][0] = m_TranslationHistory[i][0];
        m_TranslationHistory[i + 1][1] = m_TranslationHistory[i][1];
        m_TranslationHistory[i + 1][2] = m_TranslationHistory[i][2];
    }
    // Insert the latest value to the history
    m_TranslationHistory[0][0] = t[0];
    m_TranslationHistory[0][1] = t[1];
    m_TranslationHistory[0][2] = t[2];
    // Increase number of history until it reachs to the maximum size
    m_nHistory = m_nHistory < m_HistoryTerm ? m_nHistory + 1: m_nHistory;

#if 0
    // Calcurate the latest moving average to show debug message
    Eigen::Vector3f tma;
    tma[0] = tma[1] = tma[2] = 0.0f;
    for (int i = 0; i < m_nHistory; i++) {
        tma[0] += m_TranslationHistory[i][0];
        tma[1] += m_TranslationHistory[i][1];
        tma[2] += m_TranslationHistory[i][2];
    }
    tma[0] /= (float)m_nHistory;
    tma[1] /= (float)m_nHistory;
    tma[2] /= (float)m_nHistory;
    Log::Status( "PCLObjectRecognition", "Moving Average of t: (%1.3f, %1.3f, %1.3f), term = %d\n",
            tma[0], tma[1], tma[2], m_nHistory);
#endif
}