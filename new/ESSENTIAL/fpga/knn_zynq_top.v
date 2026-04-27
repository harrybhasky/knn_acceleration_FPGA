/**
 * Top-Level Integration: AXI-Lite Slave + KNN Classifier
 * 
 * This module instantiates both the AXI-lite wrapper and the KNN classifier,
 * connecting PS→PL communication to the actual classification logic.
 * 
 * Memory Map (AXI-Lite):
 *   0x00: CONTROL_REG    (RW) - bit[0]=start, bit[1]=reset
 *   0x04: STATUS_REG     (R)  - bit[0]=busy, bit[1]=pixel_ready
 *   0x08: RESULT_REG     (R)  - bit[0]=classification
 *   0x0C: PIXEL_DATA_REG (W)  - bit[7:0]=pixel value
 *   0x10: PIXEL_COUNT_REG (R) - bit[9:0]=pixel counter
 * 
 * Data Flow:
 *   PS (Python) → AXI Master Write → AXI-Lite Slave → Pixel Stream → KNN
 *   KNN → Result → AXI-Lite Slave → AXI Master Read → PS (Python)
 */

`timescale 1ns/1ps

module knn_zynq_top #(
    parameter AXI_ADDR_WIDTH = 12,
    parameter AXI_DATA_WIDTH = 32
)(
    // Clock and Reset
    input  clk,
    input  rst_n,
    
    // AXI-Lite Interface (from PS)
    // Write Address Channel
    input  [AXI_ADDR_WIDTH-1:0]   axi_awaddr,
    input                         axi_awvalid,
    output                        axi_awready,
    
    // Write Data Channel
    input  [AXI_DATA_WIDTH-1:0]   axi_wdata,
    input  [AXI_DATA_WIDTH/8-1:0] axi_wstrb,
    input                         axi_wvalid,
    output                        axi_wready,
    
    // Write Response Channel
    output [1:0]                  axi_bresp,
    output                        axi_bvalid,
    input                         axi_bready,
    
    // Read Address Channel
    input  [AXI_ADDR_WIDTH-1:0]   axi_araddr,
    input                         axi_arvalid,
    output                        axi_arready,
    
    // Read Data Channel
    output [AXI_DATA_WIDTH-1:0]   axi_rdata,
    output [1:0]                  axi_rresp,
    output                        axi_rvalid,
    input                         axi_rready,
    
    // Debug/Monitor Outputs (optional)
    output [7:0]  debug_pixel_value,
    output [9:0]  debug_pixel_index,
    output        debug_pixel_write,
    output        debug_knn_busy
);

    // Internal signals from AXI slave
    wire [7:0]   knn_result_sig;
    wire         knn_valid_sig;
    wire         knn_start_sig;
    wire [9:0]   pixel_index_sig;
    wire [7:0]   pixel_data_sig;
    wire         pixel_write_sig;
    wire         knn_busy_sig;
    wire [2:0]   distance_metric_sig;
    
    // =====================================================
    // AXI-Lite Slave Wrapper
    // =====================================================
    
    axi_lite_slave #(
        .AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),
        .AXI_DATA_WIDTH(AXI_DATA_WIDTH)
    ) axi_slave_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // AXI Write Channels
        .awaddr(axi_awaddr),
        .awvalid(axi_awvalid),
        .awready(axi_awready),
        
        .wdata(axi_wdata),
        .wstrb(axi_wstrb),
        .wvalid(axi_wvalid),
        .wready(axi_wready),
        
        .bresp(axi_bresp),
        .bvalid(axi_bvalid),
        .bready(axi_bready),
        
        // AXI Read Channels
        .araddr(axi_araddr),
        .arvalid(axi_arvalid),
        .arready(axi_arready),
        
        .rdata(axi_rdata),
        .rresp(axi_rresp),
        .rvalid(axi_rvalid),
        .rready(axi_rready),
        
        // KNN Interface
        .knn_result(knn_result_sig),
        .knn_valid(knn_valid_sig),
        .knn_start(knn_start_sig),
        .pixel_index(pixel_index_sig),
        .pixel_data(pixel_data_sig),
        .pixel_write(pixel_write_sig),
        .knn_busy(knn_busy_sig),
        .distance_metric(distance_metric_sig)
    );
    
    // =====================================================
    // KNN Classifier Instance
    // =====================================================
    
    knn_classifier_optimized knn_inst (
        .clk(clk),
        .rst_n(rst_n),
        
        // Pixel input stream
        .pixel_data(pixel_data_sig),
        .pixel_write(pixel_write_sig),
        .pixel_index(pixel_index_sig),
        
        // Control
        .start(knn_start_sig),
        
        // Parameters
        .distance_metric(distance_metric_sig),
        
        // Output
        .classification(knn_result_sig),
        .valid(knn_valid_sig),
        .busy(knn_busy_sig)
    );
    
    // =====================================================
    // Debug Outputs
    // =====================================================
    
    assign debug_pixel_value = pixel_data_sig;
    assign debug_pixel_index = pixel_index_sig;
    assign debug_pixel_write = pixel_write_sig;
    assign debug_knn_busy = knn_busy_sig;

endmodule


/**
 * Alternative: AXI with Data Movement (if using AXI DMA)
 * 
 * For higher throughput, can use AXI DMA to stream pixels directly
 * instead of single-pixel writes. This version shown for reference.
 */

module knn_zynq_top_dma #(
    parameter AXI_ADDR_WIDTH = 32,
    parameter AXI_DATA_WIDTH = 32
)(
    input clk,
    input rst_n,
    
    // AXI-Lite Control Interface
    input  [AXI_ADDR_WIDTH-1:0]   axi_awaddr,
    input                         axi_awvalid,
    output                        axi_awready,
    
    input  [AXI_DATA_WIDTH-1:0]   axi_wdata,
    input  [AXI_DATA_WIDTH/8-1:0] axi_wstrb,
    input                         axi_wvalid,
    output                        axi_wready,
    
    output [1:0]                  axi_bresp,
    output                        axi_bvalid,
    input                         axi_bready,
    
    input  [AXI_ADDR_WIDTH-1:0]   axi_araddr,
    input                         axi_arvalid,
    output                        axi_arready,
    
    output [AXI_DATA_WIDTH-1:0]   axi_rdata,
    output [1:0]                  axi_rresp,
    output                        axi_rvalid,
    input                         axi_rready,
    
    // AXI Stream Interface (for DMA pixel input)
    input  [31:0]  axi_stream_tdata,
    input          axi_stream_tvalid,
    output         axi_stream_tready,
    input          axi_stream_tlast
);
    
    // This version would use AXI Stream slave to accept pixel data
    // from DMA controller, enabling true streaming performance
    // Implementation left as reference for high-throughput variant
    
endmodule
