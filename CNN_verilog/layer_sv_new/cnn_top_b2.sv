module conv_top_b2(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in,
    input logic valid_in,
    input logic signed[15:0] weight_b1 [0:3][0:8],
    input logic signed[31:0] bias_b1 [0:3],
    input logic signed[15:0] weight_b2 [0:3][0:3][0:8],
    input logic signed[31:0] bias_b2 [0:3],
    output logic signed[15:0] pixel_out [0:3],
    output logic ready,
    output logic[3:0] valid_out
);

logic signed[15:0] pixel_out_b1 [0:3];
logic[3:0] valid_out_b1;
logic ready_b1;

logic signed[15:0] pixel_out_bridge12 [0:3];
logic full_bridge12;
logic empty_bridge12;

logic ready_b2;

assign ready = ready_b1 && !full_bridge12;

layer_1 layer_b1(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_in),
    .valid_in(valid_in),
    .weight_b1(weight_b1),
    .bias_b1(bias_b1),
    .ready(ready_b1),
    .pixel_out(pixel_out_b1),
    .valid_out(valid_out_b1)
);

sync_fifo bridge_12(
    .clk(clk),
    .reset(reset),
    .wr_en(valid_out_b1[0]),
    .rd_en(ready_b2),
    .data_in(pixel_out_b1),
    .full(full_bridge12),
    .empty(empty_bridge12),
    .data_out(pixel_out_bridge12)
);

layer_2 layer_b2(
    .clk(clk),
    .reset(reset),
    .pixel_in(pixel_out_bridge12),
    .valid_in(!empty_bridge12),
    .weight_b2(weight_b2),
    .bias_b2(bias_b2),
    .pixel_out(pixel_out),
    .ready(ready_b2),
    .valid_out(valid_out)
);




endmodule