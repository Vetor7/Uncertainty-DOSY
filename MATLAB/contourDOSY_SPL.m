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

    if Type == "FPH"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(2, [3, 1])
        DiffCoef = [4.0, 6.5, 13.8];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);

        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 
        
        text(5, 4, "Fructose", FontSize=8, Color=[1, 0.07, 0.07])
        text(5, 6.5, "Propan-1-ol", FontSize=8, Color=[1, 0.07, 0.07])
        text(4, 13.8, "HDO", FontSize=8, Color=[1, 0.07, 0.07])
        text(1, 4, "TSP", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "AMO"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(3, [3, 1])
        DiffCoef = [2.9];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 40);
        
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 

        text(6.0, 2.9, "amodiaquine", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

    elseif Type == "AD"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        nexttile(4, [3, 1])
        DiffCoef = [2.9, 3.5];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole, 50);

        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 

        text(6.0, 2.9, "amodiaquine", FontSize=8, Color=[1, 0.07, 0.07])
        text(6.0, 3.5, "desethylamodiaquine quinoneimine", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');
    
   elseif Type == "QG"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % Uncertanty
        nexttile(2, [3, 1])
        DiffCoef = [3.2, 5.4];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 

        text(7, 3.2, "quinine", FontSize=8, Color=[1, 0.07, 0.07])
        text(7, 5.4, "geraniol", FontSize=8, Color=[1, 0.07, 0.07])

        for i = 1:length(DiffCoef)
            l = line(gca,get(gca,'xlim'),DiffCoef(i)*ones(1,2),'LineWidth',0.8,'color',[0.85 0.85 0.85],'LineStyle','--');
            uistack(l, "bottom");
        end
        set(gca,'YTick',unique(DiffCoef) );

        xlabel(t, 'Chemical Shift(ppm)');
        ylabel(t, 'Diffusion Coefficient(10^{-10}m^2/s)');

   elseif Type == "MPD"
        plot(ppm,cs_spec, "Color",'k');set(gca,'Xdir','reverse');axis off;
        xlim([cs1,cs2]);

        % Uncertanty
        nexttile(2, [3, 1])
        DiffCoef = [4.8, 6.0,9.1, 12.1];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 

        text(2.5, 4.8, "N,N-dimethylethylamine", FontSize=8, Color=[1, 0.07, 0.07])
        text(2.5, 6, "N-propanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(2.5, 9.1, "Methanol", FontSize=8, Color=[1, 0.07, 0.07])
        text(2, 12.1, "D2O", FontSize=8, Color=[1, 0.07, 0.07])

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
        nexttile(11, [3, 1])
        DiffCoef = [8.3, 10.9];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);
        
        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 

        text(3.3, 8.3, "n-butyl alcohol", FontSize=8, Color=[1, 0.07, 0.07])
        text(3.3, 10.9, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])

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
        nexttile(12, [3, 1])
        DiffCoef = [7.7, 8.15, 10.6];
        contour(ppm,decay_range*(0.8/b(end)),spec_whole,40);

        xlim([cs1,cs2]);
        ylim([dc1,dc2]);

        box off;
        xLimits = xlim;
        yLimits = ylim;
        line([xLimits(1), xLimits(2)], [yLimits(1), yLimits(1)], 'Color', 'k', 'LineWidth', 1.5);
        line([xLimits(1), xLimits(1)], [yLimits(1), yLimits(2)], 'Color', 'k', 'LineWidth', 1.5);
        set(gca,'Ydir','reverse','Xdir','reverse','TickDir','out', 'LineWidth', 1.5); 

        text(3.3, 7.7, "n-butyric acid", FontSize=8, Color=[1, 0.07, 0.07])
        text(3.3, 8.1, "n-butyl alcohol", FontSize=8, Color=[1, 0.07, 0.07])
        text(3.3, 10.6, "ethanol", FontSize=8, Color=[1, 0.07, 0.07])

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
    if Type == "AMO"
        nexttile(3, [3, 1])
    elseif Type == "AD"
        nexttile(4, [3, 1])
    elseif Type == "BPP1"
        nexttile(11, [3, 1])
    elseif Type == "BPP2"
        nexttile(12, [3, 1])
    else
        nexttile(2, [3, 1])
    end
    low = 0;
    total = 0;
    for idx = 1:length(merged_boxes)
        bb = merged_boxes(idx, :, :);
        diff = max_var(idx);
        if diff == 0
            continue;
        end
        total = total + 1;
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
            low = low + 1;
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