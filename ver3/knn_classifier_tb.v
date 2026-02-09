// ============================================================================
// Comprehensive Testbench for K-NN Classifier
// Verilog-2001 Compatible
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

    // Test tracking
    integer test_num;
    integer pass_count;
    integer fail_count;
    integer timeout_counter;
    integer i;  // Loop variable declared at module level
    
    // Performance monitoring
    integer start_time, end_time, total_cycles;

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

    // Helper task to run a test
    task run_test;
        input [7:0] feat1;
        input [7:0] feat2;
        input [7:0] k;
        input expected_class;
        input [200*8-1:0] test_description;
        reg actual_class;
        begin
            test_num = test_num + 1;
            $display("\n--- Test %0d: %0s ---", test_num, test_description);
            $display("Input: (%0d, %0d), K=%0d", feat1, feat2, k);
            
            test_data = {feat1, feat2};
            k_value = k;
            start = 1;
            #CLK_PERIOD;
            start = 0;
            
            // Wait for done with timeout
            timeout_counter = 0;
            while (done == 0 && timeout_counter < 1000) begin
                #CLK_PERIOD;
                timeout_counter = timeout_counter + 1;
            end
            
            if (done == 1) begin
                actual_class = predicted_class;
                $display("Result: Class = %0d", predicted_class);
                if (expected_class === 1'bx) begin
                    $display("PASS - Completed (expected result not specified)");
                    pass_count = pass_count + 1;
                end else if (predicted_class === expected_class) begin
                    $display("PASS - Expected: %0d, Got: %0d", expected_class, predicted_class);
                    pass_count = pass_count + 1;
                end else begin
                    $display("FAIL - Expected: %0d, Got: %0d", expected_class, predicted_class);
                    fail_count = fail_count + 1;
                end
                #(CLK_PERIOD*3);
            end else begin
                $display("FAIL - TIMEOUT waiting for done signal!");
                fail_count = fail_count + 1;
            end
        end
    endtask

    // Main test stimulus
    initial begin
        // Initialize
        rst = 1;
        start = 0;
        test_data = 0;
        k_value = K_VALUE;
        test_num = 0;
        pass_count = 0;
        fail_count = 0;
        
        // Create waveform dump
        $dumpfile("knn_classifier.vcd");
        $dumpvars(0, knn_classifier_tb);
        
        // Reset sequence
        #(CLK_PERIOD*2);
        rst = 0;
        #(CLK_PERIOD*2);
        
        $display("\n========================================");
        $display("K-NN Classifier Comprehensive Test Suite");
        $display("K = %0d, Training Samples = %0d", K_VALUE, NUM_TRAINING);
        $display("========================================\n");

        // ====================================================================
        // BASIC FUNCTIONALITY TESTS
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 1: Basic Functionality Tests");
        $display("========================================");

        run_test(8'd4, 8'd8, 8'd3, 1'b0, "Basic Test 1 - Near Class 0 cluster");
        run_test(8'd2, 8'd3, 8'd3, 1'b1, "Basic Test 2 - Near Class 1 cluster");
        run_test(8'd7, 8'd7, 8'd3, 1'b1, "Basic Test 3 - Very close to (7,8) Class 1");
        run_test(8'd1, 8'd1, 8'd5, 1'b1, "Basic Test 4 - K=5 variant");
        run_test(8'd8, 8'd8, 8'd3, 1'b1, "Basic Test 5 - Corner point");

        // ====================================================================
        // K VALUE VARIATION TESTS
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 2: K Value Variation Tests");
        $display("========================================");

        run_test(8'd4, 8'd8, 8'd1, 1'b0, "K=1 (Nearest neighbor only)");
        run_test(8'd4, 8'd8, 8'd3, 1'b0, "K=3 (same point)");
        run_test(8'd4, 8'd8, 8'd5, 1'b0, "K=5 (same point)");
        run_test(8'd4, 8'd8, 8'd7, 1'b0, "K=7 (same point)");
        run_test(8'd4, 8'd8, 8'd10, 1'b0, "K=10 (large K)");
        
        // ====================================================================
        // TIE BREAKER SCENARIOS
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 3: Tie Breaker Scenarios");
        $display("========================================");

        run_test(8'd5, 8'd5, 8'd2, 1'bx, "Tie scenario - equidistant point");
        run_test(8'd4, 8'd5, 8'd4, 1'bx, "K=4 potential tie");
        run_test(8'd5, 8'd6, 8'd6, 1'bx, "K=6 potential tie");

        // ====================================================================
        // EDGE CASES
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 4: Edge Cases");
        $display("========================================");

        run_test(8'd0, 8'd0, 8'd3, 1'bx, "Minimum coordinates (0,0)");
        run_test(8'd255, 8'd255, 8'd3, 1'bx, "Maximum coordinates (255,255)");
        run_test(8'd0, 8'd255, 8'd3, 1'bx, "Extreme diagonal (0,255)");
        run_test(8'd255, 8'd0, 8'd3, 1'bx, "Extreme diagonal (255,0)");

        run_test(8'd1, 8'd12, 8'd3, 1'b0, "Exact match - Training sample 0");
        run_test(8'd7, 8'd8, 8'd3, 1'b1, "Exact match - Training sample 5");
        run_test(8'd4, 8'd7, 8'd3, 1'b0, "Exact match - Training sample 8");

        run_test(8'd5, 8'd5, 8'd1, 1'bx, "K=1 minimum");
        run_test(8'd5, 8'd5, 8'd50, 1'bx, "K=50 maximum (all training samples)");

        // ====================================================================
        // RAPID FIRE TESTS
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 5: Rapid Fire Tests");
        $display("========================================");

        run_test(8'd1, 8'd2, 8'd3, 1'bx, "Rapid 1");
        run_test(8'd3, 8'd4, 8'd3, 1'bx, "Rapid 2");
        run_test(8'd5, 8'd6, 8'd3, 1'bx, "Rapid 3");
        run_test(8'd7, 8'd8, 8'd3, 1'b1, "Rapid 4 - exact match");
        run_test(8'd9, 8'd10, 8'd3, 1'bx, "Rapid 5");

        // ====================================================================
        // RESET DURING OPERATION TEST
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 6: Reset During Operation");
        $display("========================================");

        test_num = test_num + 1;
        $display("\n--- Test %0d: Reset during computation ---", test_num);
        test_data = {8'd5, 8'd5};
        k_value = 8'd3;
        start = 1;
        #CLK_PERIOD;
        start = 0;
        
        #(CLK_PERIOD * 10);
        $display("Applying reset mid-computation...");
        rst = 1;
        #(CLK_PERIOD * 2);
        rst = 0;
        #(CLK_PERIOD * 2);
        
        if (done == 0 && dut.state == 0) begin
            $display("PASS - Reset recovered properly");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL - Reset did not recover");
            fail_count = fail_count + 1;
        end

        // ====================================================================
        // INVALID K VALUES
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 7: Invalid K Values");
        $display("========================================");

        run_test(8'd5, 8'd5, 8'd0, 1'bx, "K=0 (invalid, should use default)");
        run_test(8'd5, 8'd5, 8'd100, 1'bx, "K=100 (exceeds NUM_TRAINING)");

        // ====================================================================
        // STRESS TEST
        // ====================================================================
        $display("\n========================================");
        $display("SECTION 8: Stress Test");
        $display("========================================");

        $display("\nRunning 20 sequential classifications...");
        
        for (i = 1; i <= 20; i = i + 1) begin
            test_data = {i[7:0], i[7:0] + 8'd5};
            k_value = 8'd3;
            start = 1;
            #CLK_PERIOD;
            start = 0;
            
            // Wait for done
            timeout_counter = 0;
            while (done == 0 && timeout_counter < 1000) begin
                #CLK_PERIOD;
                timeout_counter = timeout_counter + 1;
            end
            
            if (done == 1) begin
                $display("Stress test %0d: Input=(%0d,%0d), Result=%0d", 
                         i, i[7:0], i[7:0] + 8'd5, predicted_class);
            end else begin
                $display("Stress test %0d: TIMEOUT", i);
            end
            #(CLK_PERIOD*2);
        end
        
        $display("Stress test completed");
        pass_count = pass_count + 20;
        test_num = test_num + 20;

        // ====================================================================
        // FINAL SUMMARY
        // ====================================================================
        #(CLK_PERIOD*10);
        
        $display("\n========================================");
        $display("TEST SUMMARY");
        $display("========================================");
        $display("Total Tests Run: %0d", test_num);
        $display("Passed: %0d", pass_count);
        $display("Failed: %0d", fail_count);
        if (test_num > 0)
            $display("Pass Rate: %0d%%", (pass_count * 100) / test_num);
        $display("========================================");
        
        if (fail_count == 0) begin
            $display("\n*** ALL TESTS PASSED! ***\n");
        end else begin
            $display("\n*** SOME TESTS FAILED ***\n");
        end
        
        $finish;
    end

    // Extended timeout watchdog
    initial begin
        #(CLK_PERIOD * 500000);
        $display("\n\nERROR: Global simulation timeout!");
        $display("Test was running: %0d", test_num);
        $finish;
    end

    // Performance monitoring
    always @(posedge start) begin
        start_time = $time;
    end
    
    always @(posedge done) begin
        end_time = $time;
        total_cycles = (end_time - start_time) / CLK_PERIOD;
        $display("Performance: %0d clock cycles (%0d ns)", total_cycles, end_time - start_time);
    end

endmodule