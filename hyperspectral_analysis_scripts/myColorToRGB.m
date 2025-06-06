%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Helper function: Convert color strings to RGB triples
function rgb = myColorToRGB(colorStr)
    if colorStr(1) == '#'
        % Remove '#' and convert hex to RGB
        hexStr = colorStr(2:end);
        if numel(hexStr) ~= 6
            error('Invalid hex color: %s', colorStr);
        end
        r = double(hex2dec(hexStr(1:2))) / 255;
        g = double(hex2dec(hexStr(3:4))) / 255;
        b = double(hex2dec(hexStr(5:6))) / 255;
        rgb = [r, g, b];
    else
        % Convert MATLAB shorthand colors to RGB
        switch lower(colorStr)
            case 'r'
                rgb = [1, 0, 0];
            case 'g'
                rgb = [0, 1, 0];
            case 'b'
                rgb = [0, 0, 1];
            case 'c'
                rgb = [0, 1, 1];
            case 'm'
                rgb = [1, 0, 1];
            case 'y'
                rgb = [1, 1, 0];
            case 'k'
                rgb = [0, 0, 0];
            case 'w'
                rgb = [1, 1, 1];
            otherwise
                error('Unknown color string: %s', colorStr);
        end
    end
end