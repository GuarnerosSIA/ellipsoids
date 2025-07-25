figure
hold on

[X,Z] = max(result.data.f');

for i = 1:9
    plot(data.X(Z==i),data.X(Z==i,2));
end