/**
 * AXI-Lite Slave Wrapper for KNN Classifier
 * 
 * Memory-Mapped Register Interface:
 * ================================
 * 0x00: CONTROL_REG
 *       [0]    = start (write 1 to start processing)
 *       [1]    = reset (write 1 to reset)
 *       [7:4]  = distance_metric (0=Manhattan, 1=Euclidean)
 *
 * 0x04: STATUS_REG (read-only)
 *       [0]    = busy (1=processing, 0=done)
 *       [1]    = pixel_ready (1=ready for next pixel)
 *       [7:4]  = reserved
 *
 * 0x08: RESULT_REG (read-only)
 *       [0]    = classification (0=benign, 1=malignant)
 *       [7:1]  = reserved
 *
 * 0x0C: PIXEL_DATA_REG (write-only)
 *       [7:0]  = 8-bit pixel value
 *       Write one pixel at a time; auto-increments pixel counter
 *
 * 0x10: PIXEL_COUNT_REG (read-only)
 *       [9:0]  = current pixel count (0-783)
 *
 * Data Flow:
 * 1. PS writes start=1 to 0x00 (CONTROL_REG)
 * 2. For each pixel (784 total):
 *    - Wait until pixel_ready=1 in STATUS_REG
 *    - Write 8-bit value to PIXEL_DATA_REG
 * 3. Poll STATUS_REG until busy=0
 * 4. Read classification from RESULT_REG
 */

`timescale 1ns/1ps

module axi_lite_slave #(
    parameter AXI_ADDR_WIDTH = 12,
    parameter AXI_DATA_WIDTH = 32
)(
    // AXI Clock & Reset
    input clk,
    input rst_n,
    
    // AXI-Lite Write Address Channel
    input  [AXI_ADDR_WIDTH-1:0] awaddr,
    input                       awvalid,
    output                      awready,
    
    // AXI-Lite Write Data Channel
    input  [AXI_DATA_WIDTH-1:0] wdata,
    input  [AXI_DATA_WIDTH/8-1:0] wstrb,
    input                       wvalid,
    output                      wready,
    
    // AXI-Lite Write Response Channel
    output [1:0]                bresp,
    output                      bvalid,
    input                       bready,
    
    // AXI-Lite Read Address Channel
    input  [AXI_ADDR_WIDTH-1:0] araddr,
    input                       arvalid,
    output                      arready,
    
    // AXI-Lite Read Data Channel
    output [AXI_DATA_WIDTH-1:0] rdata,
    output [1:0]                rresp,
    output                      rvalid,
    input                       rready,
    
    // KNN Classifier Interface
    input  [7:0]  knn_result,           // Classification result from KNN
    input         knn_valid,            // KNN has valid result
    output        knn_start,            // Start signal to KNN
    output [9:0]  pixel_index,          // Pixel number being processed
    output [7:0]  pixel_data,           // Current pixel value
    output        pixel_write,          // Write enable for pixel
    output        knn_busy,             // KNN module busy status
    output [2:0]  distance_metric       // 0=Manhattan, 1=Euclidean
);

    // ===========================================
    // Register Declarations
    // ===========================================
    
    reg [31:0]  control_reg;              // 0x00
    reg [31:0]  status_reg;               // 0x04
    reg [31:0]  result_reg;               // 0x08
    reg [9:0]   pixel_count;              // Internal pixel counter (0-783)
    
    reg         knn_processing;           // Flag: KNN currently processing
    reg         pixel_ready_flag;         // Flag: Ready for next pixel
    
    // AXI handshake signals
    reg         write_addr_ready;
    reg         write_data_ready;
    reg         write_resp_valid;
    reg         read_addr_ready;
    reg         read_data_valid;
    
    // ===========================================
    // Output Assignments
    // ===========================================
    
    assign awready = write_addr_ready;
    assign wready  = write_data_ready;
    assign bresp   = 2'b00;               // OKAY response
    assign bvalid  = write_resp_valid;
    
    assign arready = read_addr_ready;
    assign rresp   = 2'b00;               // OKAY response
    assign rvalid  = read_data_valid;
    
    // KNN interface
    assign knn_start        = control_reg[0];
    assign knn_busy         = knn_processing;
    assign distance_metric  = control_reg[7:4];
    assign pixel_index      = pixel_count;
    assign pixel_write      = (control_reg[0] & pixel_ready_flag);
    
    // ===========================================
    // AXI Write Address Channel (Simple)
    // ===========================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_addr_ready <= 1'b1;
        end else begin
            // Always ready for write addresses
            write_addr_ready <= 1'b1;
        end
    end
    
    // ===========================================
    // AXI Write Data Channel with Register File
    // ===========================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_data_ready <= 1'b1;
            control_reg <= 32'b0;
            pixel_count <= 10'b0;
        end else if (wvalid & wready) begin
            write_data_ready <= 1'b1;
            
            // Decode address from last write address transaction
            // In real design, would store awaddr and use it here
            case (awaddr[5:2])  // Word-aligned address
                4'h0: begin  // 0x00 - CONTROL_REG
                    control_reg <= wdata;
                    // Reset pixel counter on start
                    if (wdata[0]) pixel_count <= 10'b0;
                end
                
                4'h3: begin  // 0x0C - PIXEL_DATA_REG (write-only)
                    pixel_data <= wdata[7:0];
                    // Auto-increment pixel counter
                    if (pixel_count < 10'd783) begin
                        pixel_count <= pixel_count + 1;
                    end
                end
                
                default: begin
                    // Other registers are read-only
                end
            endcase
        end else begin
            write_data_ready <= wvalid;
        end
    end
    
    // ===========================================
    // AXI Write Response Channel
    // ===========================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            write_resp_valid <= 1'b0;
        end else if (wvalid & wready & !write_resp_valid) begin
            write_resp_valid <= 1'b1;
        end else if (bready) begin
            write_resp_valid <= 1'b0;
        end
    end
    
    // ===========================================
    // AXI Read Address Channel
    // ===========================================
    
    reg [AXI_ADDR_WIDTH-1:0] read_addr;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_addr_ready <= 1'b1;
            read_addr <= {AXI_ADDR_WIDTH{1'b0}};
        end else if (arvalid & read_addr_ready) begin
            read_addr <= araddr;
            read_addr_ready <= 1'b0;
        end else if (!read_data_valid) begin
            read_addr_ready <= 1'b1;
        end
    end
    
    // ===========================================
    // AXI Read Data Channel with Register File
    // ===========================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            read_data_valid <= 1'b0;
        end else if (arvalid & read_addr_ready) begin
            read_data_valid <= 1'b1;
        end else if (rready) begin
            read_data_valid <= 1'b0;
        end
    end
    
    // Multiplex read data based on address
    reg [31:0] read_data_mux;
    
    always @(*) begin
        case (read_addr[5:2])
            4'h0:    read_data_mux = control_reg;           // 0x00
            4'h1:    read_data_mux = status_reg;            // 0x04
            4'h2:    read_data_mux = result_reg;            // 0x08
            4'h4:    read_data_mux = {22'b0, pixel_count}; // 0x10
            default: read_data_mux = 32'b0;
        endcase
    end
    
    assign rdata = read_data_mux;
    
    // ===========================================
    // Status Register Updates
    // ===========================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            knn_processing <= 1'b0;
            pixel_ready_flag <= 1'b0;
            result_reg <= 32'b0;
            status_reg <= 32'b0;
        end else begin
            // Pixel ready when not processing or between samples
            pixel_ready_flag <= ~knn_processing | (pixel_count >= 10'd783);
            
            // Start processing when start bit written
            if (control_reg[0] & pixel_count == 10'd0) begin
                knn_processing <= 1'b1;
            end
            
            // Stop processing when KNN reports valid result
            if (knn_valid & knn_processing) begin
                knn_processing <= 1'b0;
                result_reg <= {31'b0, knn_result[0]};
            end
            
            // Update status register
            status_reg <= {30'b0, pixel_ready_flag, knn_processing};
        end
    end
    
    // ===========================================
    // Reset Logic
    // ===========================================
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            control_reg <= 32'b0;
            pixel_count <= 10'b0;
            result_reg <= 32'b0;
        end else if (control_reg[1]) begin  // Reset bit
            control_reg <= 32'b0;
            pixel_count <= 10'b0;
            result_reg <= 32'b0;
        end
    end

endmodule
