// ============================================================================
// Top Module for Basys3 FPGA Board
// K-NN Classifier Implementation
// ============================================================================

module knn_basys3_top (
    input wire clk,              // 100MHz clock from Basys3
    input wire btnC,             // Center button - reset
    input wire btnU,             // Up button - start classification
    input wire [7:0] sw,         // Switches for test data input
    input wire [1:0] sw_feature, // SW[15:14] - Feature select (00=feat1, 01=feat2)
    output wire [15:0] led,      // LEDs for output
    output wire [6:0] seg,       // 7-segment display
    output wire [3:0] an         // 7-segment anodes
);

    // Internal signals
    reg rst_sync, start_sync;
    reg [7:0] test_feature1, test_feature2;
    reg [15:0] test_data_reg;
    wire predicted_class;
    wire done;
    
    // Button debouncing registers
    reg [19:0] btn_counter_rst, btn_counter_start;
    reg btnC_prev, btnU_prev;
    reg btnC_debounced, btnU_debounced;
    
    // State for input capture
    reg [1:0] input_state;
    localparam INPUT_FEATURE1 = 2'd0;
    localparam INPUT_FEATURE2 = 2'd1;
    localparam INPUT_READY = 2'd2;
    localparam INPUT_PROCESSING = 2'd3;
    
    // Debounce center button (reset)
    always @(posedge clk) begin
        btnC_prev <= btnC;
        if (btnC && !btnC_prev) begin
            btn_counter_rst <= 20'd0;
        end else if (btn_counter_rst < 20'd1000000) begin
            btn_counter_rst <= btn_counter_rst + 1;
        end
        
        if (btn_counter_rst == 20'd999999) begin
            btnC_debounced <= 1;
        end else begin
            btnC_debounced <= 0;
        end
    end
    
    // Debounce up button (start)
    always @(posedge clk) begin
        btnU_prev <= btnU;
        if (btnU && !btnU_prev) begin
            btn_counter_start <= 20'd0;
        end else if (btn_counter_start < 20'd1000000) begin
            btn_counter_start <= btn_counter_start + 1;
        end
        
        if (btn_counter_start == 20'd999999) begin
            btnU_debounced <= 1;
        end else begin
            btnU_debounced <= 0;
        end
    end
    
    // Input state machine
    always @(posedge clk) begin
        if (btnC_debounced) begin
            input_state <= INPUT_FEATURE1;
            test_feature1 <= 8'd0;
            test_feature2 <= 8'd0;
            rst_sync <= 1;
            start_sync <= 0;
        end else begin
            rst_sync <= 0;
            
            case (input_state)
                INPUT_FEATURE1: begin
                    if (btnU_debounced) begin
                        test_feature1 <= sw;
                        input_state <= INPUT_FEATURE2;
                    end
                end
                
                INPUT_FEATURE2: begin
                    if (btnU_debounced) begin
                        test_feature2 <= sw;
                        input_state <= INPUT_READY;
                        test_data_reg <= {test_feature1, sw};
                    end
                end
                
                INPUT_READY: begin
                    if (btnU_debounced) begin
                        start_sync <= 1;
                        input_state <= INPUT_PROCESSING;
                    end
                end
                
                INPUT_PROCESSING: begin
                    start_sync <= 0;
                    if (done) begin
                        input_state <= INPUT_FEATURE1;
                    end
                end
            endcase
        end
    end
    
    // Instantiate KNN classifier
    knn_classifier #(
        .DATA_WIDTH(8),
        .NUM_FEATURES(2),
        .NUM_TRAINING(50),
        .K_VALUE(3)
    ) knn_inst (
        .clk(clk),
        .rst(rst_sync),
        .start(start_sync),
        .test_data(test_data_reg),
        .k_value(8'd3),
        .predicted_class(predicted_class),
        .done(done)
    );
    
    // LED outputs
    assign led[7:0] = test_feature1;     // Show feature 1
    assign led[15:8] = test_feature2;    // Show feature 2
    assign led[0] = (input_state == INPUT_FEATURE1);  // Waiting for feature 1
    assign led[1] = (input_state == INPUT_FEATURE2);  // Waiting for feature 2
    assign led[2] = (input_state == INPUT_READY);     // Ready to classify
    assign led[3] = (input_state == INPUT_PROCESSING); // Processing
    assign led[14] = done;               // Classification done
    assign led[15] = predicted_class;    // Predicted class (0 or 1)
    
    // 7-segment display controller
    reg [31:0] display_counter;
    reg [1:0] digit_select;
    reg [3:0] digit_value;
    
    always @(posedge clk) begin
        if (btnC_debounced) begin
            display_counter <= 0;
            digit_select <= 0;
        end else begin
            if (display_counter >= 100000) begin
                display_counter <= 0;
                digit_select <= digit_select + 1;
            end else begin
                display_counter <= display_counter + 1;
            end
        end
    end
    
    // Select which digit to display
    always @(*) begin
        case (digit_select)
            2'd0: digit_value = test_feature1[3:0];    // Feature 1 low nibble
            2'd1: digit_value = test_feature1[7:4];    // Feature 1 high nibble
            2'd2: digit_value = test_feature2[3:0];    // Feature 2 low nibble
            2'd3: digit_value = test_feature2[7:4];    // Feature 2 high nibble
            default: digit_value = 4'd0;
        endcase
    end
    
    // Anode control (active low)
    assign an = ~(4'b0001 << digit_select);
    
    // 7-segment decoder
    seven_segment_decoder seg_decoder (
        .digit(digit_value),
        .segments(seg)
    );

endmodule

// ============================================================================
// 7-Segment Display Decoder
// ============================================================================
module seven_segment_decoder (
    input wire [3:0] digit,
    output reg [6:0] segments  // {g, f, e, d, c, b, a}
);

    always @(*) begin
        case (digit)
            4'h0: segments = 7'b1000000; // 0
            4'h1: segments = 7'b1111001; // 1
            4'h2: segments = 7'b0100100; // 2
            4'h3: segments = 7'b0110000; // 3
            4'h4: segments = 7'b0011001; // 4
            4'h5: segments = 7'b0010010; // 5
            4'h6: segments = 7'b0000010; // 6
            4'h7: segments = 7'b1111000; // 7
            4'h8: segments = 7'b0000000; // 8
            4'h9: segments = 7'b0010000; // 9
            4'hA: segments = 7'b0001000; // A
            4'hB: segments = 7'b0000011; // b
            4'hC: segments = 7'b1000110; // C
            4'hD: segments = 7'b0100001; // d
            4'hE: segments = 7'b0000110; // E
            4'hF: segments = 7'b0001110; // F
            default: segments = 7'b1111111; // blank
        endcase
    end

endmodule
