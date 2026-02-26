module classifier(
    input logic clk,
    input logic reset,
    input logic signed[15:0] pixel_in [0:3],
    input logic valid_in,
    input logic signed[15:0] weight [0:9][0:3][0:48],
    input logic signed[31:0] bias [0:9],
    output logic[3:0] result,
    output logic done
);

logic[2:0] state, next_state;
localparam CALCULATE = 0, MAXING = 1, FINISH = 2;

integer i, j;
logic signed[47:0] acc [0:9];
logic signed[47:0] current_max;
logic[3:0] current_result;
logic[5:0] count;

always_ff @(posedge clk or posedge reset) begin
    if(reset) state <= CALCULATE;
    else state <= next_state;
end

always_comb begin
    case(state)
        CALCULATE: begin
            if(count==48) next_state = MAXING;
            else next_state = state;
        end
        MAXING: begin
            if(count==9) next_state = FINISH;
            else next_state = state;
        end
        default: next_state = state;
    endcase
end

assign done = (state==FINISH);

//count
always_ff @(posedge clk or posedge reset) begin
    if(reset) count <= 0;
    else begin
        case(state)
            CALCULATE: begin
                if(count==48) count <= 0;
                else if(valid_in) count <= count + 1;
            end
            MAXING: begin
                count <= count + 1;
            end
            default: count <= 0;
        endcase
    end
end

//acc
always_ff @(posedge clk or posedge reset) begin
    if(reset) begin
        for(i=0; i<10; i=i+1) acc[i] <= bias[i];
    end else begin
        if(state==CALCULATE && valid_in) begin
            for(i=0; i<10; i=i+1) begin
                acc[i] <= acc[i] + (pixel_in[0] * weight[i][0][count] +
                                    pixel_in[1] * weight[i][1][count] +
                                    pixel_in[2] * weight[i][2][count] +
                                    pixel_in[3] * weight[i][3][count]);
            end
        end
    end
end

//current_max & current_result
always_ff @(posedge clk) begin
    if(state==MAXING) begin
        if(count==0) begin
            current_max <= acc[0];
            result <= 0;
        end else begin
            if(current_max < acc[count]) begin
                current_max <= acc[count];
                result <= count;
            end
        end
    end
end


endmodule