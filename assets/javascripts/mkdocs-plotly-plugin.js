'use strict'

const plotlyCharts = [...document.querySelectorAll("div.mkdocs-plotly")].map(d => {
    return {div: d}
});
const bodyelement = document.querySelector('body');

const defaultTemplate = document.querySelector("#default-template-settings");
const slateTemplate = document.querySelector("#slate-template-settings");

const defaultTemlateSettings = defaultTemplate ? JSON.parse(defaultTemplate.textContent) : null;
const slateTemplateSettings = slateTemplate ? JSON.parse(slateTemplate.textContent) : null;


function updateTemplates() {
    if (bodyelement.getAttribute('data-md-color-scheme') == 'slate') {
        plotlyCharts.forEach(chart => {
            if (!chart.div.classList.contains("no-auto-theme") && chart.load)
                Plotly.relayout(chart.div, chart.slateTemplate)
        });
    } else { 
        plotlyCharts.forEach(chart => {
            if (!chart.div.classList.contains("no-auto-theme") && chart.load)
                Plotly.relayout(chart.div, chart.defaultTemlate)
        });
    }
}
function updateTemplate(chart) {
    if (bodyelement.getAttribute('data-md-color-scheme') == 'slate') {
        if (!(chart.div).classList.contains("no-auto-theme") && chart.load)
            Plotly.relayout(chart.div, chart.slateTemplate)
    } else { 
        if (!chart.div.classList.contains("no-auto-theme") && chart.load)
            Plotly.relayout(chart.div, chart.defaultTemlate)
    }
}

if (slateTemplateSettings && defaultTemlateSettings) {
    const observer = new MutationObserver(mutations => {
        mutations.forEach(mutation => {
            if (mutation.type === "attributes") {
                if (mutation.attributeName == "data-md-color-scheme") {
                    updateTemplates();
                }
            }
        });
    });
    observer.observe(bodyelement, {
        attributes: true //configure it to listen to attribute changes
    });
}


async function fetchData(url) {
    const resp = await fetch(url);
    const data = await resp.json();
    return data;
}

plotlyCharts.forEach(chart => {
    if (chart.div.dataset.jsonpath)
        fetchData(chart.div.dataset.jsonpath).then(
            plot_data => {
                const data = plot_data.data ? plot_data.data : {};
                const layout = plot_data.layout ? plot_data.layout : {};
                const config = plot_data.config ? plot_data.config : {};
                Plotly.newPlot(chart.div, data, layout, config);
                chart.load = true;
                chart.slateTemplate = plot_data.slateTemplate ? plot_data.slateTemplate : slateTemplateSettings;
                chart.defaultTemlate = plot_data.defaultTemplate ? plot_data.defaultTemplate : defaultTemlateSettings;
                updateTemplate(chart);
            }
        )
    else {
        const plot_data = JSON.parse(chart.div.textContent);
        chart.div.textContent = '';
        const data = plot_data.data ? plot_data.data : {};
        const layout = plot_data.layout ? plot_data.layout : {};
        const config = plot_data.config ? plot_data.config : {};
        Plotly.newPlot(chart.div, data, layout, config);
        chart.load = true;
        chart.slateTemplate = plot_data.slateTemplate ? plot_data.slateTemplate : slateTemplateSettings;
        chart.defaultTemlate = plot_data.defaultTemplate ? plot_data.defaultTemplate : defaultTemlateSettings;
        updateTemplate(chart);
    }
})



