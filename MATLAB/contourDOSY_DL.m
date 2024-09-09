function contourDOSY_DL(Type, Result_DRILT, Result, HNMR, b, ppm, ...
    idx_peaks_DRILT, idx_peaks, cs1, cs2, dc1, dc2, t, ...
    max_var, merged_boxes)
    cs_spec = zeros([length(ppm), 1]);
    cs_spec(idx_peaks, :) = HNMR;
    spec_DRILT = zeros([length(Result_DRILT(1, :)), length(ppm)]);
    if idx_peaks_DRILT ~= 0
        spec_DRILT(:, idx_peaks_DRILT) = Result_DRILT.';
    end
    DRILT_decay_range = linspace(0, (length(Result_DRILT(1, :))-1)/10, length(Result_DRILT(1, :)));
    spec_whole = zeros([length(Result(1, :)), length(ppm)]);
    spec_whole(:, idx_peaks) = Result.';
    decay_range = linspace(0, (length(Result(1, :))-1)/10, length(Result(1, :)));

    if Type == "GSP"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % DRILT
        nexttile(2, [3, 1])
        DiffCoef = [2.21, 3.02, 4.14];
        contour(ppm,DRILT_decay_range*(0.8/b(end)),spec_DRILT, 40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.4, 1.9, '(a)DRECT', FontSize=8)
        text(5.0, 2.16, "PEG600", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 2.97, "sucrose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5.0, 4.09, "glucose", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );
        set(gca,'xtick',[])
        
        % DReaM
        nexttile(5, [3, 1])
        DiffCoef = [2.17, 3.05, 4.24];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.4, 1.9, '(b)DReaM', FontSize=8)
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
        
    elseif Type == "QDC"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(2, [3, 1])
        DiffCoef = [4.6, 8.1, 11.0];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "TSP"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(2, [3, 1])
        DiffCoef = [4.0, 6.5, 13.8];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "EC"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(2, [3, 1])
        DiffCoef = [2.9];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "AMDK"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(2, [3, 1])
        DiffCoef = [2.9, 3.5];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

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

        % DRILT
        nexttile(2, [3, 1])
        DiffCoef = [4.6, 7.2, 10.3];
        contour(ppm,DRILT_decay_range*(0.8/b(end)),spec_DRILT,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(12.2, 3, '(a)DRECT', FontSize=8)
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
        nexttile(5, [3, 1])
        DiffCoef = [4.6, 7.2, 10.4];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(12.2, 3, '(b)DReaM', FontSize=8)
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

        % DRILT
        nexttile(2, [3, 1])
        DiffCoef = [3.2, 4.3, 5.0, 6.3, 8.2, 10.5];
        contour(ppm,DRILT_decay_range*(1.2/b(end)),spec_DRILT,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.3, 2.5, '(a)DRECT', FontSize=8)
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
        nexttile(5, [3, 1])
        DiffCoef = [3.2, 4.3, 5.0, 6.3, 8.3, 10.8];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(5.3, 2.5, '(b)DReaM', FontSize=8)
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
        set(gca,'YTick',unique(DiffCoef));

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');
    
   elseif Type == "QG"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % Uncertanty
        nexttile(2, [3, 1])
        DiffCoef = [3.2, 5.4];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(7, 3.2, "quinine", FontSize=8, Color=[1, 0.07, 0.07])
        text(7, 5.4, "geraniol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

   elseif Type == "JNN"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % Uncertanty
        nexttile(2, [3, 1])
        DiffCoef = [4.8, 6.0,9.1, 12.1];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "BPP1"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % Uncertainty
        nexttile(2, [3, 1])
        DiffCoef = [8.2, 11.1];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(4, 8.2, "n-butyl alcohol", FontSize=8, Color=[1, 0.07, 0.07])
        text(4, 11.1, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef));

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');
        
    elseif Type == "BPP2"

        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % Uncertainty
        nexttile(2, [3, 1])
        DiffCoef = [7.7, 8.1, 10.6];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);
        set(gca,'Ydir','reverse','Xdir','reverse'); 
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);
        text(4, 7.7, "n-butyric acid", FontSize=8, Color=[1, 0.07, 0.07])
        text(4, 8.1, "n-butyl alcohol", FontSize=8, Color=[1, 0.07, 0.07])
        text(4, 10.6, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef));

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    end

    % Display uncertainty
    hold on; 
    yIndices = 1:length(decay_range);
    decay_range = double(decay_range);
    b = double(b);
    if Type == "QGC"||Type == "GSP"||Type == "M6"
        nexttile(5, [3, 1])
    else
        nexttile(2, [3, 1])
    end
    if Type == "BPP1"
        ppm = ppm(400:1000);
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
        if diff <= 0.5
            color = [0 0.4 0]; % RGB for green 
            symbols = '✓';
            fontsize = 8;

        else
            color = 'r'; % Red 
            symbols = '?';
            fontsize = 8;
        end

        x_rect = [x1Mapped-0.001*horizC, x2Mapped+0.001*horizC, x2Mapped+0.001*horizC, x1Mapped-0.001*horizC];
        y_rect = [y1Mapped, y1Mapped, y2Mapped, y2Mapped];
        h_rect = fill(x_rect, y_rect, color,'EdgeColor', 'none', 'FaceAlpha', 0.3); 

        uistack(h_rect, 'bottom');
        text(centreX, y1Mapped - 0.05 * horizD, symbols, 'Color', color, 'FontSize', fontsize, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');

    end
    hold off;

end