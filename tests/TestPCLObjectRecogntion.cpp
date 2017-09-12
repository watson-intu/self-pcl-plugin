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


#include "utils/UnitTest.h"
#include "utils/Log.h"
#include "utils/ThreadPool.h"
#include "utils/Config.h"
#include "services/PCLObjectRecognition.h"

class TestPCLObjectRecognition : UnitTest 
{
public:
	//! Construction
	TestPCLObjectRecognition() : UnitTest("TestPCL"),
		m_bObjectRecogized(false)
	{ }

	bool	m_bObjectRecogized;

	virtual void RunTest()
	{
		Config config;
		Test(ISerializable::DeserializeFromFile("./etc/tests/unit_test_config.json", &config) != NULL);

		FILE * fp = fopen( "./etc/tests/pcl_depth_image.png", "rb" );
		Test( fp != NULL );

		fseek( fp, 0, SEEK_END );
		size_t pngSize = ftell( fp );
		fseek( fp, 0, SEEK_SET );

		std::string depthData;
		depthData.resize( pngSize );
		fread( &depthData[0], 1, pngSize, fp );
		fclose( fp );

		ThreadPool pool(5);

		PCLObjectRecognition service;
		std::vector<std::string> models;
		models.push_back( "shared/pcd/drill/d000.pcd" );
		models.push_back( "shared/pcd/drill/d045.pcd" );
		models.push_back( "shared/pcd/drill/d090.pcd" );
		models.push_back( "shared/pcd/drill/d135.pcd" );
		models.push_back( "shared/pcd/drill/d180.pcd" );
		models.push_back( "shared/pcd/drill/d225.pcd" );
		models.push_back( "shared/pcd/drill/d270.pcd" );
		models.push_back( "shared/pcd/drill/d315.pcd" );
		std::vector<PCLObjectRecognition::ObjectModel> objects;
		objects.push_back( PCLObjectRecognition::ObjectModel( "drill", models ) );

		service.SetObjects( objects );
		Test( service.Start() );

		service.ClassifyObjects( depthData, DELEGATE( TestPCLObjectRecognition, OnClassifyObjects, const Json::Value &, this ) );
		Spin( m_bObjectRecogized, 300.0f );
		Test( m_bObjectRecogized );
	}

	void OnClassifyObjects(const Json::Value & a_Json )
	{
		Test(! a_Json.isNull() );
		Log::Debug( "TestPCLObjectRecognition", "OnClassifyObjedcts: %s", a_Json.toStyledString().c_str() );
		m_bObjectRecogized = true;
	}

};

TestPCLObjectRecognition TEST_PCL;