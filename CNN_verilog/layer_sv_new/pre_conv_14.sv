module pre_conv_layer_14(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in,
    input logic valid_in,
    input logic signed[15:0] weight [0:8],
    output logic ready,
    output logic signed[47:0] pixel_out,
    output logic valid_out
);

logic signed[15:0] pad_data;
logic signed[15:0] window [0:8];
logic pad_valid;
logic lb_valid;
logic pe_valid;
logic[15:0] pe_result;
logic[9:0] pixel_now;

padding_control #(.IMG_SIZE(14)) pad(
    .clk(clk),
    .reset(reset),
    .data_in(pixel_in),
    .valid_in(valid_in),
    .ready(ready),
    .data_out(pad_data),
    .valid_out(pad_valid)
);

line_buffer #(.IMG_SIZE(16)) lb(
    .clk(clk),
    .reset(reset),
    .pixel_in(pad_data),
    .valid_in(pad_valid),
    .window(window),
    .valid_out(lb_valid)
);

pre_pe_core pe(
    .clk(clk),
    .reset(reset),
    .window(window),
    .weight(weight),
    .valid_in(lb_valid),
    .data_out(pixel_out),
    .valid_out(valid_out)
);

always_ff @(posedge clk or posedge reset) begin
    if(reset) pixel_now <= 0;
    else if(valid_out) pixel_now <= pixel_now + 1;
end


endmodule