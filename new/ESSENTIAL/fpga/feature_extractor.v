// ============================================================================
// Feature Extractor Module
// Extracts 2 features from 28x28 pixel image:
//   Feature 0: Mean pixel intensity (average of all 784 pixels)
//   Feature 1: Texture metric (standard deviation or variance)
// 
// Input: 784 pixels (28x28), 8-bit each, streamed serially over 784 cycles
// Output: 2 x 8-bit features suitable for KNN classification
// ============================================================================

module feature_extractor #(
    parameter PIXEL_WIDTH    = 8,
    parameter IMAGE_SIZE     = 28,      // 28x28 = 784 pixels
    parameter NUM_PIXELS     = 784,
    parameter ACCUMULATOR_WIDTH = 12    // Wide enough for sum: 784 * 255 = 199,920 < 2^18
)(
    input  wire                          clk,
    input  wire                          rst,
    input  wire                          pixel_valid,
    input  wire [PIXEL_WIDTH-1:0]        pixel_in,
    output reg  [PIXEL_WIDTH-1:0]        feature0,      // Mean intensity
    output reg  [PIXEL_WIDTH-1:0]        feature1,      // Variance/texture
    output reg                           features_valid,
    output reg  [15:0]                   pixel_count    // Debug: count pixels received
);

    // ── State Encoding ──────────────────────────────────────────────────
    localparam IDLE          = 3'd0;
    localparam LOAD_PIXELS   = 3'd1;
    localparam CALC_MEAN     = 3'd2;
    localparam CALC_VARIANCE = 3'd3;
    localparam OUTPUT        = 3'd4;

    reg [2:0] state, next_state;
    
    // ── Accumulation Registers ──────────────────────────────────────────
    reg [17:0] sum_pixels;              // Sum of all pixel values
    reg [25:0] sum_squares;             // Sum of squared pixel values (for variance)
    reg [15:0] pixel_idx;               // Current pixel index
    
    reg [PIXEL_WIDTH-1:0] mean_value;   // Calculated mean
    reg [PIXEL_WIDTH-1:0] variance_value; // Calculated variance
    
    // ── Pipeline Registers ──────────────────────────────────────────────
    reg [PIXEL_WIDTH-1:0] pixel_reg;
    
    // ── Sequential Logic ────────────────────────────────────────────────
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            sum_pixels <= 0;
            sum_squares <= 0;
            pixel_idx <= 0;
            pixel_count <= 0;
            feature0 <= 0;
            feature1 <= 0;
            features_valid <= 0;
            mean_value <= 0;
            variance_value <= 0;
        end else begin
            state <= next_state;
            
            case (state)
                IDLE: begin
                    sum_pixels <= 0;
                    sum_squares <= 0;
                    pixel_idx <= 0;
                    pixel_count <= 0;
                    features_valid <= 0;
                end
                
                LOAD_PIXELS: begin
                    if (pixel_valid && pixel_idx < NUM_PIXELS) begin
                        pixel_reg <= pixel_in;
                        sum_pixels <= sum_pixels + {{6{1'b0}}, pixel_in};
                        sum_squares <= sum_squares + {pixel_in, 2'b00} * pixel_in;  // pixel^2
                        pixel_idx <= pixel_idx + 1;
                        pixel_count <= pixel_count + 1;
                    end
                end
                
                CALC_MEAN: begin
                    mean_value <= sum_pixels / NUM_PIXELS;  // 199920 / 784 = 255 max
                end
                
                CALC_VARIANCE: begin
                    // Variance = E[X^2] - E[X]^2
                    // mean_sq = sum_squares / NUM_PIXELS
                    // variance = mean_sq - mean_value^2
                    variance_value <= (sum_squares / NUM_PIXELS) - (mean_value * mean_value);
                end
                
                OUTPUT: begin
                    feature0 <= mean_value;
                    feature1 <= variance_value;
                    features_valid <= 1;
                end
            endcase
        end
    end
    
    // ── Combinational Next-State Logic ──────────────────────────────────
    always @(*) begin
        next_state = state;
        case (state)
            IDLE:          if (pixel_valid) next_state = LOAD_PIXELS;
            LOAD_PIXELS:   if (pixel_idx >= NUM_PIXELS) next_state = CALC_MEAN;
            CALC_MEAN:                     next_state = CALC_VARIANCE;
            CALC_VARIANCE:                 next_state = OUTPUT;
            OUTPUT:                        next_state = IDLE;
            default:                       next_state = IDLE;
        endcase
    end

endmodule
