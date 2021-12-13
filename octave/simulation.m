epsilon = 1e-3
x1 = [1 2;3 4;5 6]
w1 = [1 1 1 2 2 2 3;-3 -2 -2 -2 -1 -1 -1]
z1 = x1*w1 
a1 = z1 .* (z1>=0)
d1 = (z1>=0)
w2 = [1;2;1;-1;-2;-1;7]
z2 = a1 * w2
a2 = z2 .* (z2>=0)
d2 = (z2>=0)

y= [8;8;8]


function cost = cost(w_1,w_2)
    x = [1 2;3 4;5 6];
    y= [8;8;8];
    z_1=x*w_1;
    a_1 = z_1 .* (z_1>=0);
    z_2 = a_1 * w_2;
    a_2 = z_2 .* (z_2>=0);
    cost = sqrt(mean((y-a_2).^2));
endfunction

C = sqrt(mean((y-a2).^2))

delta_3 = (1/(3*C)) * ((y-a2).*d2)

dJdw2 = transpose(a1) * delta_3

w2_t = transpose(w2)
w1_t = transpose(w1)

delta_2 = (delta_3 * w2_t) .* d1

dJdw1 = transpose(x1) * delta_2;

edJd1 = zeros(2,7);
for i = 1:2
    for j = 1:7
        p = zeros(2,7);
        p(i,j)=epsilon;
        loss1 = cost(w1+p,w2);
        loss2 = cost(w1-p,w2);
        estimation = (loss2 - loss1) / (2*epsilon);
        edJd1(i,j) = estimation;
    endfor
endfor

dJdw1
edJd1

edJd2 = zeros(7,1);
for i = 1:7
    for j = 1:1
        p = zeros(7,1);
        p(i,j)=epsilon;
        loss1 = cost(w1,w2+p);
        loss2 = cost(w1,w2-p);
        estimation = (loss2 - loss1) / (2*epsilon);
        edJd2(i,j) = estimation;
    endfor
endfor

dJdw2
edJd2