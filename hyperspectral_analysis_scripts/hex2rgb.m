function rgb = hex2rgb(hexStr)
    if hexStr(1) == '#'
        hexStr = hexStr(2:end);
    end
    r = double(hex2dec(hexStr(1:2)))/255;
    g = double(hex2dec(hexStr(3:4)))/255;
    b = double(hex2dec(hexStr(5:6)))/255;
    rgb = [r, g, b];
end