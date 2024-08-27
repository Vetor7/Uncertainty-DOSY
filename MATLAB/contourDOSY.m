function contourDOSY(Type, Result_LRsPILT, Result_CoMeF, Result_DRILT, ...
    Result, HNMR, b, ppm, cs, idx_peaks_DRILT, idx_peaks, cs1, cs2, ...
    dc1, dc2, t, var_spec, max_var, merged_boxes)

    cs_spec = zeros([length(ppm), 1]);
    cs_spec(idx_peaks, :) = HNMR;

    spec_DRILT = zeros([length(Result_DRILT(1, :)), length(ppm)]);
    spec_DRILT(:, idx_peaks_DRILT) = Result_DRILT.';
    DRILT_decay_range = linspace(0, (length(Result_DRILT(1, :))-1)/10, length(Result_DRILT(1, :)));
    
    spec_whole = zeros([length(Result(1, :)), length(ppm)]);
    spec_whole(:, idx_peaks) = Result.';
    decay_range = linspace(0, (length(Result(1, :))-1)/10, length(Result(1, :)));



    if  Type == "GSP"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % LRsPLIT
        nexttile(2, [2, 1])
        DiffCoef = [2.2, 3.12, 4.02];
        contour(ppm,linspace(0, 7, 351), Result_LRsPILT, 20);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.4, 1.9, 'a', FontSize=12)
        text(5.0, 2.2, "PEG600", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 3.07, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 3.97, "glucose", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        
        % CoMeF
        nexttile(4, [2, 1])
        DiffCoef = [2.2, 3.11, 4.01];
        contour(ppm,linspace(1.5, 5, 100), Result_CoMeF, 20);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.4, 1.9, 'b', FontSize=12)
        text(5.0, 2.2, "PEG600", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 3.07, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 3.97, "glucose", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        
        % DRILT
        nexttile(6, [2, 1])
        DiffCoef = [2.21, 3.02, 4.14];
        contour(ppm,DRILT_decay_range*(0.8/b(end)),spec_DRILT, 40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.4, 1.9, 'c', FontSize=12)
        text(5.0, 2.16, "PEG600", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 2.97, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 4.09, "glucose", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])
        
        % Uncertanty
        nexttile(8, [2, 1])
        DiffCoef = [2.17, 3.05, 4.24];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.4, 1.9, 'd', FontSize=12)
        text(5.0, 2.16, "PEG600", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 2.97, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 4.09, "glucose", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "QGC"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % LRsPILT
        nexttile(2, [2, 1])
        DiffCoef = [4.7, 7.3, 10.1];
        contour(ppm,linspace(1, 20, 191), Result_LRsPILT, 40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(12.2, 3, 'a', FontSize=12)
        text(10.1, 4.66, "quinine", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 7.26, "geraniol", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 10.06, "camphene", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        % CoMeF
        nexttile(4, [2, 1])
        DiffCoef = [4.6, 7.3, 9.9];
        contour(ppm,linspace(1, 12, 100), Result_CoMeF, 40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(12.2, 3, 'b', FontSize=12)
        text(10.1, 4.56, "quinine", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 7.26, "geraniol", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 9.86, "camphene", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        % DRILT
        nexttile(6, [2, 1])
        DiffCoef = [4.6, 7.2, 10.3];
        contour(ppm,DRILT_decay_range*(0.8/b(end)),spec_DRILT,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(12.2, 3, 'c', FontSize=12)
        text(10.1, 4.46, "quinine", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 7.16, "geraniol", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 10.36, "camphene", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        % Uncertanty
        nexttile(8, [2, 1])
        DiffCoef = [4.6, 7.2, 10.4];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(12.2, 3, 'd', FontSize=12)
        text(10.1, 4.46, "quinine", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 7.16, "geraniol", FontSize=8, Color=[1, 0.07, 0.07])
        text(10.1, 10.36, "camphene", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "M6"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % LRsPILT
        nexttile(2, [3, 1])
        DiffCoef = [3.2, 4.2, 5.2, 6.3, 8.2, 10.6];
        contour(ppm,linspace(1, 20, 191), Result_LRsPILT, 40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.3, 3, 'a', FontSize=12)
        text(5, 3.16, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.16, "lysine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 5.16, "threonine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 6.26, "butanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 8.16, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 10.56, "methanol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        % CoMeF
        nexttile(5, [3, 1])
        DiffCoef = [3.1, 3.2, 3.3, 4.2, 4.3, 4.9, 6.2, 8.2, 10.7];
        contour(cs,linspace(2, 12, 100), Result_CoMeF, 20);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.3, 3, 'b', FontSize=12)
        text(5, 3.1, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.16, "lysine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.86, "threonine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 6.16, "butanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 8.16, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 10.66, "methanol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        % DRILT
        nexttile(8, [3, 1])
        DiffCoef = [3.2, 4.3, 5.0, 6.3, 8.2, 10.5];
        contour(ppm,DRILT_decay_range*(1.2/b(end)),spec_DRILT,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.3, 2.5, 'c', FontSize=12)
        text(5, 3.16, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.26, "lysine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.96, "threonine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 6.26, "butanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 8.16, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 10.7, "methanol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])

        % Uncertainty
        nexttile(11, [3, 1])
        DiffCoef = [3.2, 4.3, 5.0, 6.3, 8.3, 10.8];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.3, 2.5, 'd', FontSize=12)
        text(5, 3.16, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.26, "lysine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 4.96, "threonine", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 6.26, "butanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 8.16, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 10.7, "methanol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    end

    hold on; 
    yIndices = 1:length(decay_range);
    decay_range = double(decay_range);
    b = double(b);
    if Type == "M6"
        nexttile(11, [3, 1])
    else 
        nexttile(8, [2, 1])
    end

    for idx = 1:length(merged_boxes)
        bb = merged_boxes(idx, :, :);
        diff = max_var(idx);
        if diff == 0
            continue;
        end
        x1 = bb(1, 1, 1)+1;
        x2 = bb(1, 2, 1)+1;
        y1 = bb(1, 1, 2)+1;
        y2 = bb(1, 2, 2)+1;
        try
            x1Mapped = ppm(1, x1);
            x2Mapped = ppm(1, x2);
            y1Mapped = interp1(double(yIndices), decay_range * (0.8 / b(1, end)), double(y1), 'linear', 'extrap');
            y2Mapped = interp1(double(yIndices), decay_range * (0.8 / b(1, end)), double(y2), 'linear', 'extrap');
        catch e
            fprintf('Error during interpolation: %s\n', e.message);
            continue;
        end
    
        centreX = (x2Mapped + x1Mapped) / 2;
        horizD = dc2 - dc1; 
        horizC = cs2 - cs1;
        if diff <= 0.45
            color = [0 0.4 0]; % RGB for green 
            symbols = '✓';
            fontsize = 8;

        else
            color = 'r'; % Red 
            symbols = '▲';
            fontsize = 8;
        end


        x_rect = [x1Mapped-0.001*horizC, x2Mapped+0.001*horizC, x2Mapped+0.001*horizC, x1Mapped-0.001*horizC];
        y_rect = [y1Mapped, y1Mapped, y2Mapped, y2Mapped];
        h_rect = fill(x_rect, y_rect, color,'EdgeColor', 'none', 'FaceAlpha', 0.3); % FaceAlpha设置透明度

        uistack(h_rect, 'bottom');
        text(centreX, y1Mapped - 0.05 * horizD, symbols, 'Color', color, 'FontSize', fontsize, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    end
    hold off;

end