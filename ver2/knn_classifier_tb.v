// ============================================================================
// Testbench for K-NN Classifier (Fast Version)
// ============================================================================

`timescale 1ns / 1ps

module knn_classifier_tb;

    // Parameters
    parameter DATA_WIDTH = 8;
    parameter NUM_FEATURES = 2;
    parameter NUM_TRAINING = 50;
    parameter K_VALUE = 3;
    parameter CLK_PERIOD = 10;

    // Testbench signals
    reg clk;
    reg rst;
    reg start;
    reg [DATA_WIDTH*NUM_FEATURES-1:0] test_data;
    reg [DATA_WIDTH-1:0] k_value;
    wire predicted_class;
    wire done;

    // Instantiate the DUT (Device Under Test)
    knn_classifier #(
        .DATA_WIDTH(DATA_WIDTH),
        .NUM_FEATURES(NUM_FEATURES),
        .NUM_TRAINING(NUM_TRAINING),
        .K_VALUE(K_VALUE)
    ) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .test_data(test_data),
        .k_value(k_value),
        .predicted_class(predicted_class),
        .done(done)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // Test stimulus
    integer test_num;
    initial begin
        // Initialize signals
        rst = 1;
        start = 0;
        test_data = 0;
        k_value = K_VALUE;
        test_num = 0;
        
        // Create waveform dump
        $dumpfile("knn_classifier.vcd");
        $dumpvars(0, knn_classifier_tb);
        
        // Wait for reset
        #(CLK_PERIOD*2);
        rst = 0;
        #(CLK_PERIOD*2);
        
        $display("\n========================================");
        $display("K-NN Classifier Testbench");
        $display("K = %0d, Training Samples = %0d", K_VALUE, NUM_TRAINING);
        $display("========================================\n");

        // Test Case 1: Test point (4, 8)
        test_num = 1;
        $display("Test %0d: Input = (4, 8)", test_num);
        test_data = {8'd4, 8'd8};
        k_value = 3;
        start = 1;
        #CLK_PERIOD;
        start = 0;
        
        wait(done == 1);
        $display("Result: Class = %0d\n", predicted_class);
        #(CLK_PERIOD*5);

        // Test Case 2: Test point (2, 3)
        test_num = 2;
        $display("Test %0d: Input = (2, 3)", test_num);
        test_data = {8'd2, 8'd3};
        k_value = 3;
        start = 1;
        #CLK_PERIOD;
        start = 0;
        
        wait(done == 1);
        $display("Result: Class = %0d\n", predicted_class);
        #(CLK_PERIOD*5);

        // Test Case 3: Test point (7, 7)
        test_num = 3;
        $display("Test %0d: Input = (7, 7)", test_num);
        test_data = {8'd7, 8'd7};
        k_value = 3;
        start = 1;
        #CLK_PERIOD;
        start = 0;
        
        wait(done == 1);
        $display("Result: Class = %0d\n", predicted_class);
        #(CLK_PERIOD*5);

        // Test Case 4: Test point (1, 1) with K=5
        test_num = 4;
        $display("Test %0d: Input = (1, 1), K = 5", test_num);
        test_data = {8'd1, 8'd1};
        k_value = 5;
        start = 1;
        #CLK_PERIOD;
        start = 0;
        
        wait(done == 1);
        $display("Result: Class = %0d\n", predicted_class);
        #(CLK_PERIOD*5);

        // Test Case 5: Test point (8, 8)
        test_num = 5;
        $display("Test %0d: Input = (8, 8)", test_num);
        test_data = {8'd8, 8'd8};
        k_value = 3;
        start = 1;
        #CLK_PERIOD;
        start = 0;
        
        wait(done == 1);
        $display("Result: Class = %0d\n", predicted_class);
        #(CLK_PERIOD*10);

        $display("\n========================================");
        $display("All tests completed!");
        $display("========================================\n");
        
        $finish;
    end

    // Timeout watchdog
    initial begin
        #(CLK_PERIOD * 100000);
        $display("\nERROR: Simulation timeout!");
        $finish;
    end

    // Monitor for debugging - updated for fast version
    initial begin
        $monitor("Time=%0t | State=%0d | i=%0d | k_counter=%0d | Done=%b | Class=%b", 
                 $time, dut.state, dut.i_reg, dut.k_counter, done, predicted_class);
    end

endmodule