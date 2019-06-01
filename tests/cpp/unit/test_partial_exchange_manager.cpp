#include "testing.hpp"

TEST(partial_exchange_manager_test_1, test_bin_packing)
{
    partial_exchange_manager manager;


    std::vector<int> steps = {1};
    std::vector<float> fractions = {0.5};
    manager.addSchedule(steps, fractions);
    manager.addGlobalTensorInfo(3, 20);

    manager.addTensorInfo("first", 10);
    manager.addTensorInfo("second", 5);
    manager.addTensorInfo("third", 5);

    std::vector<partition> partitions = manager.repartition(1).partitions_;
    
    ASSERT_EQ(partitions.size(), 2);

    ASSERT_EQ(partitions[0].index, 1);
    ASSERT_TRUE(partitions[0].tensorNames.find("first") != partitions[0].tensorNames.end());   

    ASSERT_EQ(partitions[1].index, 2);
    ASSERT_TRUE(partitions[1].tensorNames.find("second") != partitions[1].tensorNames.end());    
    ASSERT_TRUE(partitions[1].tensorNames.find("third") != partitions[1].tensorNames.end());    
}


TEST(partial_exchange_manager_test_2, test_bin_packing)
{
    partial_exchange_manager manager;
    // 4, 8, 1, 4, 2, 1, total=20 
    std::vector<int> steps = {1};
    std::vector<float> fractions = {0.5};
    manager.addSchedule(steps, fractions);
    manager.addGlobalTensorInfo(6, 20);

    manager.addTensorInfo("four:1", 4);
    manager.addTensorInfo("eight:1", 8);
    manager.addTensorInfo("one:1", 1);
    manager.addTensorInfo("four:2", 4);
    manager.addTensorInfo("two:1", 2);
    manager.addTensorInfo("one:2", 1);

    std::vector<partition> partitions = manager.repartition(1).partitions_;
    
    ASSERT_EQ(partitions.size(), 2);

    ASSERT_EQ(partitions[0].index, 1);
    // 8, 2
    ASSERT_TRUE(partitions[0].tensorNames.find("eight:1") != partitions[0].tensorNames.end());    
    ASSERT_TRUE(partitions[0].tensorNames.find("two:1") != partitions[0].tensorNames.end());    

    ASSERT_EQ(partitions[1].index, 2);
    // 4, 4, 2
    ASSERT_TRUE(partitions[1].tensorNames.find("four:1") != partitions[1].tensorNames.end());   
    ASSERT_TRUE(partitions[1].tensorNames.find("four:2") != partitions[1].tensorNames.end());   
    ASSERT_TRUE(partitions[1].tensorNames.find("one:1") != partitions[1].tensorNames.end());   
    ASSERT_TRUE(partitions[1].tensorNames.find("one:2") != partitions[1].tensorNames.end());   
}

TEST(partial_exchange_manager_test_3, test_bin_packing)
{
    partial_exchange_manager manager;
    // 9, 8, 2, 2, 5, 4, total=30
    // capacity = 10
    std::vector<int> steps = {1};
    std::vector<float> fractions = {0.34};
    manager.addSchedule(steps, fractions);
    manager.addGlobalTensorInfo(6, 30);


    manager.addTensorInfo("nine:1", 9);
    manager.addTensorInfo("eight:1", 8);
    manager.addTensorInfo("two:1", 2);
    manager.addTensorInfo("two:2", 2);
    manager.addTensorInfo("five:1", 5);
    manager.addTensorInfo("four:1", 4);

    std::vector<partition> partitions = manager.repartition(1).partitions_;
    
    for(auto p : partitions)
        std::cout << p << std::endl;

    ASSERT_EQ(partitions.size(), 4);

    ASSERT_EQ(partitions[0].index, 1);
    // 9
    ASSERT_TRUE(partitions[0].tensorNames.find("nine:1") != partitions[0].tensorNames.end());    


    ASSERT_EQ(partitions[1].index, 2);
    // 8, 2
    ASSERT_TRUE(partitions[1].tensorNames.find("eight:1") != partitions[1].tensorNames.end());   
    ASSERT_TRUE(partitions[1].tensorNames.find("two:2") != partitions[1].tensorNames.end());   


    ASSERT_EQ(partitions[2].index, 3);
    // 4, 5
    ASSERT_TRUE(partitions[2].tensorNames.find("four:1") != partitions[2].tensorNames.end());   
    ASSERT_TRUE(partitions[2].tensorNames.find("five:1") != partitions[2].tensorNames.end());   


    ASSERT_EQ(partitions[3].index, 4);
    // 4, 5
    ASSERT_TRUE(partitions[3].tensorNames.find("two:1") != partitions[3].tensorNames.end());   
  
}


