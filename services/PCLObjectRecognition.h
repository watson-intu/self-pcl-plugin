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


#ifndef PLC_OBJECT_RECOGNITION_H
#define PLC_OBJECT_RECOGNITION_H

#include "services/IObjectRecognition.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#ifdef VISUALIZATION
#include <pcl/visualization/pcl_visualizer.h>
#endif

class PCLObjectRecognition : public IObjectRecognition
{
public:
    RTTI_DECL();

    //! Types
    typedef Delegate<const Json::Value&>   OnClassifyObjects;

    typedef pcl::PointXYZ       PointType;

    //! This structure is used to hold loaded point cloud data for a particular angle of a model
    struct ModelPCD
    {
        ModelPCD() :
            m_Model( new pcl::PointCloud<PointType>() )
        {}

        pcl::PointCloud<PointType>::Ptr m_Model;
    };

    struct ObjectModel : public ISerializable
    {
        ObjectModel()
        {}
        ObjectModel( const std::string& a_ObjectId, const std::vector<std::string>& a_Models ) :
            m_ObjectId( a_ObjectId ), m_Models( a_Models ) , m_nHistory ( 0 )
        {}

        std::string                     m_ObjectId;     // the ID of this object
        std::vector<std::string>        m_Models;       // list of files containing the PCD
        std::vector<ModelPCD>           m_PCD;          // loaded point-cloud data

        int                             m_HistoryTerm;  // History term to calculate moving average
        int                             m_nHistory;     // number of moving average histories
        std::vector<Eigen::Vector3f>    m_TranslationHistory; // historical data of position

        //! ISerializable interface
        virtual void Serialize(Json::Value& json)
        {
            json["m_ObjectId"] = m_ObjectId;
            SerializeVector( "m_Models", m_Models, json );
        }
        virtual void Deserialize(const Json::Value& json)
        {
            if ( json["m_ObjectId"].isString() )
            {
                m_ObjectId = json["m_ObjectId"].asString();
            }

            DeserializeVector( "m_Models", json, m_Models );
        }
        virtual void setHistoryTerm(int term)
        {
            m_HistoryTerm = term;
            m_TranslationHistory.resize(m_HistoryTerm);
        }

        bool LoadPCD();
        float getMovingAveDivergence(Eigen::Vector3f t);
        void updateMovingAve(Eigen::Vector3f t);
    };

    //! Construction
    PCLObjectRecognition();

    //! ISerializable interface
    virtual void Serialize(Json::Value& json);
    virtual void Deserialize(const Json::Value& json);

    //! IService interface
    virtual bool Start();

    //! IObjectRecognition interface
    virtual void ClassifyObjects(const std::string& a_DepthImageData,
                                 OnClassifyObjects a_Callback );

    void SetObjects( const std::vector<ObjectModel>& a_Objects )
    {
        m_Objects = a_Objects;
    }

private:
    //! Types
    struct ProcessDepthData
    {
        ProcessDepthData( const std::string& a_DepthData, OnClassifyObjects a_Callback ) :
            m_DepthData( a_DepthData ), m_Callback( a_Callback )
        {}

        std::string         m_DepthData;
        Json::Value         m_Results;
        OnClassifyObjects   m_Callback;
    };

    void ProcessThread( ProcessDepthData* a_pData );
    void SendResults( ProcessDepthData* a_pData );

    //! Data
    std::vector<ObjectModel>        m_Objects;
    float                           m_ClusterSizeMin1; // Minimum number of points for clustering stage 1
    float                           m_ClusterSizeMax1; // Maximum number of points for clustering stage 1
    float                           m_ClusterSizeMin2; // Minimum number of points for clustering stage 2
    float                           m_ClusterSizeMax2; // Maximum number of points for clustering stage 2
    float                           m_ClusterTolerance1; // Distance tolerance between points in clustering stage 1
    float                           m_ClusterTolerance2; // Distance tolerance between points in clustering stage 2
    float                           m_AcceptableFittingScore; // Maximum ICP fitting score to accept (lower score is better)
    float                           m_DivRange; // Maximum acceptable divergence from moving average
    int                             m_MeanK; // Coefficient to change smoothing level
    float                           m_SamplingSize; // Sampling distance to reduce point cloud
    int                             m_HistoryTerm; // History term to calculate moving average

#ifdef VISUALIZATION
    pcl::visualization::PCLVisualizer *viewer; // visualization window handler
#endif
};

#endif
