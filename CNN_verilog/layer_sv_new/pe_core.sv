module pe_core (    //PROCESSING ELEMENT
    input logic clk,
    input logic reset,  //active high
    input logic signed[15:0] window [0:8],   //3x3 window
    input logic signed[15:0] weight [0:8],
    input logic signed[31:0] bias,
    input logic valid_in,
    output logic signed[15:0] data_out,
    output logic valid_out
);

//---MULTI REG---
// 0 1 2 (TOP)
// 3 4 5 (MOD)
// 6 7 8 (BOT)

logic signed[31:0] multi [0:8];
logic signed[47:0] sum;
logic signed[47:0] shifted;
logic valid_wait;

//multipliers
always_comb begin
    multi[0] = window[0] * weight[0];
    multi[1] = window[1] * weight[1];
    multi[2] = window[2] * weight[2];
    multi[3] = window[3] * weight[3];
    multi[4] = window[4] * weight[4];
    multi[5] = window[5] * weight[5];
    multi[6] = window[6] * weight[6];
    multi[7] = window[7] * weight[7];
    multi[8] = window[8] * weight[8];
end


//sum
always_comb begin
    sum = multi[0] + multi[1] + multi[2] + multi[3] + multi[4] + multi[5] + multi[6] + multi[7] + multi[8] + bias;
end

//shifted
assign shifted = sum >>> 14;    //fix point


//data_out & valid
always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        valid_out <= 0;
        data_out <= 0;
    end else begin
        valid_wait <= valid_in;
        valid_out <= valid_wait;
        if(valid_wait) begin
            if(shifted < 0) data_out <= 16'd0;
            else if(shifted > 32767) data_out <= 16'd32767;
            else data_out <= shifted[15:0];
        end
    end
end


endmodule