module layer_3(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in [0:3],
    input logic valid_in,
    input logic signed[15:0] weight_b3 [0:3][0:3][0:8],
    input logic signed[31:0] bias_b3 [0:3],
    output logic signed[15:0] pixel_out [0:3],
    output logic ready,
    output logic[3:0] valid_out
);

logic signed[47:0] data_out [0:3][0:3];
logic signed[47:0] sum [0:3];
logic signed[47:0] shifted [0:3];
logic[3:0] valid_channels[0:3];
logic[15:0] ready_channels;
logic[9:0] pixel_now [0:3];

assign sum[0] = data_out[0][0] + data_out[1][0] + data_out[2][0] + data_out[3][0] + bias_b3[0];
assign sum[1] = data_out[0][1] + data_out[1][1] + data_out[2][1] + data_out[3][1] + bias_b3[1];
assign sum[2] = data_out[0][2] + data_out[1][2] + data_out[2][2] + data_out[3][2] + bias_b3[2];
assign sum[3] = data_out[0][3] + data_out[1][3] + data_out[2][3] + data_out[3][3] + bias_b3[3];
assign shifted[0] = sum[0] >>> 14;
assign shifted[1] = sum[1] >>> 14;
assign shifted[2] = sum[2] >>> 14;
assign shifted[3] = sum[3] >>> 14;
assign valid_out[0] = (valid_channels[0]==4'b1111);
assign valid_out[1] = (valid_channels[1]==4'b1111);
assign valid_out[2] = (valid_channels[2]==4'b1111);
assign valid_out[3] = (valid_channels[3]==4'b1111);
assign ready = (ready_channels==16'b1111_1111_1111_1111);

always_comb begin
    if(shifted[0] > 32767) pixel_out[0] = 16'd32767;
    else if(shifted[0] < 0) pixel_out[0] = 0;
    else pixel_out[0] = shifted[0][15:0];

    if(shifted[1] > 32767) pixel_out[1] = 16'd32767;
    else if(shifted[1] < 0) pixel_out[1] = 0;
    else pixel_out[1] = shifted[1][15:0];

    if(shifted[2] > 32767) pixel_out[2] = 16'd32767;
    else if(shifted[2] < 0) pixel_out[2] = 0;
    else pixel_out[2] = shifted[2][15:0];

    if(shifted[3] > 32767) pixel_out[3] = 16'd32767;
    else if(shifted[3] < 0) pixel_out[3] = 0;
    else pixel_out[3] = shifted[3][15:0];
end

genvar i, j;
generate
    for(i=0; i<4; i=i+1) begin
        for(j=0; j<4; j=j+1) begin
            pre_conv_layer_14 channel(
                .clk(clk),
                .reset(reset),
                .pixel_in(pixel_in[i]),
                .valid_in(valid_in),
                .weight(weight_b3[j][i]),
                .ready(ready_channels[j+i*4]),
                .pixel_out(data_out[i][j]),
                .valid_out(valid_channels[j][i])
            );
        end
    end
endgenerate

// pre_conv_layer_28 channel_1_1(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[0]),
//     .valid_in(valid_in),
//     .weight(weight_b2[0][0]),
//     .ready(ready_channels[0]),
//     .pixel_out(data_out[0][0]),
//     .valid_out(valid_channels[0][0])
// );

// pre_conv_layer_28 channel_1_2(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[0]),
//     .valid_in(valid_in),
//     .weight(weight_b2[1][0]),
//     .ready(ready_channels[1]),
//     .pixel_out(data_out[0][1]),
//     .valid_out(valid_channels[1][0])
// );

// pre_conv_layer_28 channel_1_3(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[0]),
//     .valid_in(valid_in),
//     .weight(weight_b2[2][0]),
//     .ready(ready_channels[2]),
//     .pixel_out(data_out[0][2]),
//     .valid_out(valid_channels[2][0])
// );

// pre_conv_layer_28 channel_1_4(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[0]),
//     .valid_in(valid_in),
//     .weight(weight_b2[3][0]),
//     .ready(ready_channels[3]),
//     .pixel_out(data_out[0][3]),
//     .valid_out(valid_channels[3][0])
// );

// pre_conv_layer_28 channel_2_1(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[1]),
//     .valid_in(valid_in),
//     .weight(weight_b2[0][1]),
//     .ready(ready_channels[4]),
//     .pixel_out(data_out[1][0]),
//     .valid_out(valid_channels[0][1])
// );

// pre_conv_layer_28 channel_2_2(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[1]),
//     .valid_in(valid_in),
//     .weight(weight_b2[1][1]),
//     .ready(ready_channels[5]),
//     .pixel_out(data_out[1][1]),
//     .valid_out(valid_channels[1][1])
// );

// pre_conv_layer_28 channel_2_3(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[1]),
//     .valid_in(valid_in),
//     .weight(weight_b2[2][1]),
//     .ready(ready_channels[6]),
//     .pixel_out(data_out[1][2]),
//     .valid_out(valid_channels[2][1])
// );

// pre_conv_layer_28 channel_2_4(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[1]),
//     .valid_in(valid_in),
//     .weight(weight_b2[3][1]),
//     .ready(ready_channels[7]),
//     .pixel_out(data_out[1][3]),
//     .valid_out(valid_channels[3][1])
// );

// pre_conv_layer_28 channel_3_1(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[2]),
//     .valid_in(valid_in),
//     .weight(weight_b2[0][2]),
//     .ready(ready_channels[8]),
//     .pixel_out(data_out[2][0]),
//     .valid_out(valid_channels[0][2])
// );

// pre_conv_layer_28 channel_3_2(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[2]),
//     .valid_in(valid_in),
//     .weight(weight_b2[1][2]),
//     .ready(ready_channels[9]),
//     .pixel_out(data_out[2][1]),
//     .valid_out(valid_channels[1][2])
// );

// pre_conv_layer_28 channel_3_3(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[2]),
//     .valid_in(valid_in),
//     .weight(weight_b2[2][2]),
//     .ready(ready_channels[10]),
//     .pixel_out(data_out[2][2]),
//     .valid_out(valid_channels[2][2])
// );

// pre_conv_layer_28 channel_3_4(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[2]),
//     .valid_in(valid_in),
//     .weight(weight_b2[3][2]),
//     .ready(ready_channels[11]),
//     .pixel_out(data_out[2][3]),
//     .valid_out(valid_channels[3][2])
// );

// pre_conv_layer_28 channel_4_1(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[3]),
//     .valid_in(valid_in),
//     .weight(weight_b2[0][3]),
//     .ready(ready_channels[12]),
//     .pixel_out(data_out[3][0]),
//     .valid_out(valid_channels[0][3])
// );

// pre_conv_layer_28 channel_4_2(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[3]),
//     .valid_in(valid_in),
//     .weight(weight_b2[1][3]),
//     .ready(ready_channels[13]),
//     .pixel_out(data_out[3][1]),
//     .valid_out(valid_channels[1][3])
// );

// pre_conv_layer_28 channel_4_3(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[3]),
//     .valid_in(valid_in),
//     .weight(weight_b2[2][3]),
//     .ready(ready_channels[14]),
//     .pixel_out(data_out[3][2]),
//     .valid_out(valid_channels[2][3])
// );

// pre_conv_layer_28 channel_4_4(
//     .clk(clk),
//     .reset(reset),
//     .pixel_in(pixel_in[3]),
//     .valid_in(valid_in),
//     .weight(weight_b2[3][3]),
//     .ready(ready_channels[15]),
//     .pixel_out(data_out[3][3]),
//     .valid_out(valid_channels[3][3])
// );





always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        pixel_now[0] <= 0;
        pixel_now[1] <= 0;
        pixel_now[2] <= 0;
        pixel_now[3] <= 0;
    end else begin
        if(valid_out[0]) pixel_now[0] <= pixel_now[0] + 1;
        if(valid_out[1]) pixel_now[1] <= pixel_now[1] + 1;
        if(valid_out[2]) pixel_now[2] <= pixel_now[2] + 1;
        if(valid_out[3]) pixel_now[3] <= pixel_now[3] + 1;
    end
end



endmodule