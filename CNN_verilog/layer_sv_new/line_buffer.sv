module line_buffer #(
    parameter IMG_SIZE = 30 //image size is 28x28, but with padding = 1
)(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in,
    input logic valid_in,
    output logic signed[15:0] window [0:8],
    output logic valid_out
);

//---WINDOW REG---
// 0 1 2 (TOP)
// 3 4 5 (MOD)
// 6 7 8 (BOT) <- pixel_in

logic signed[15:0] save_0 [0:IMG_SIZE-1]; //save the tom row
logic signed[15:0] save_1 [0:IMG_SIZE-1]; //save the middle row
logic[5:0] pixel_count, row_count;


//valud_out
always_comb begin
    if(valid_in && row_count >= 2 && pixel_count >= 2) valid_out = 1;
    else valid_out = 0;
end

//counter
always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        pixel_count <= 0;
        row_count <= 0;
    end else if(valid_in) begin
        if(pixel_count==IMG_SIZE-1) begin
            pixel_count <= 0;
            if(row_count==IMG_SIZE-1) row_count <= 0;
            else row_count <= row_count + 1;
        end else pixel_count <= pixel_count + 1;
    end
end

//pixel
always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        window[0] <= 0;
        window[1] <= 0;
        window[2] <= 0;
        window[3] <= 0;
        window[4] <= 0;
        window[5] <= 0;
        window[6] <= 0;
        window[7] <= 0;
        window[8] <= 0;
    end else if(valid_in) begin
        window[0] <= window[1];
        window[1] <= window[2];
        window[2] <= save_0[pixel_count];
        window[3] <= window[4];
        window[4] <= window[5];
        window[5] <= save_1[pixel_count];
        window[6] <= window[7];
        window[7] <= window[8];
        window[8] <= pixel_in;
    end
end

//save
always_ff @(posedge clk) begin
    if(valid_in) begin
        save_0[pixel_count] <= save_1[pixel_count];
        save_1[pixel_count] <= pixel_in;
    end
end


endmodule