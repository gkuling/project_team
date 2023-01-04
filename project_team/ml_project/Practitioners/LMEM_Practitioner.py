from project_team.project_config import project_config
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.cm import get_cmap
import numpy as np
import pandas as pd
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.filterwarnings("error", category=ConvergenceWarning,
                        message='MixedLM optimization failed, trying a ' \
                                'different optimizer may help.')

class PLMEM_Practitioner_config(project_config):
    def __init__(self,
                 response='X',
                 covariate='Time',
                 random_effects_groups='MRN',
                 seperable_groups='MenopausalStatus',
                 random_slope=False,
                 uncorrelated_re_slopes=False,
                 optimizers=['bfgs', 'lbfgs', 'cg', 'basinhopping', 'powell'],
                 visualization=False,
                 vis_x=None,
                 **kwargs):
        super(PLMEM_Practitioner_config, self).__init__('ML_PTPractitioner')
        self.response = response
        self.covariate = covariate
        self.random_effects_groups = random_effects_groups
        self.seperable_groups = seperable_groups
        self.adjusted = type(covariate)==list
        self.optimizers = optimizers
        self.visualization = visualization
        self.vis_x = vis_x
        self.formula = self.response + " ~ " + ' + '.join(self.covariate) \
            if self.adjusted \
            else self.response + " ~ " + self.covariate
        self.re_formula = None if not random_slope \
            else ('~' + ' + '.join(self.covariate)
                  if self.adjusted
                  else '~' + self.covariate)
        if uncorrelated_re_slopes:
            self.free = \
                sm.regression.mixed_linear_model.MixedLMParams\
                    .from_components(
                    np.ones(len(self.covariate)+1),
                    np.eye(len(self.covariate)+1)
                ) \
                    if self.adjusted \
                    else \
                    sm.regression.mixed_linear_model.MixedLMParams\
                        .from_components(
                        np.ones(2), np.eye(2)
                    )
        else:
            self.free = None

class LMEM_Practitioner():
    def __init__(self,
                 dt_processor,
                 io_manager,
                 config = PLMEM_Practitioner_config()):
        self.config = config
        self.dt_processor = dt_processor
        self.io_manager = io_manager

    def evaluate(self, input_data_set=None):
        if input_data_set is None:
            dataset = pd.DataFrame(
                [self.dt_processor.if_dset.__getitem__(i) for i
                 in range(len(self.dt_processor.if_dset))]
            )
        else:
            dataset = input_data_set

        if self.config.adjusted:
            result = self.adjusted_evaluate(dataset)
        else:
            processed_data = dataset[[self.config.covariate,
                              self.config.random_effects_groups,
                              self.config.response]].copy(deep=True)

            result = self.run_lmem(processed_data)

            if self.config.seperable_groups:
                group_names = list(set(dataset[
                                           self.config.seperable_groups
                                       ].to_list()))
                for nm in group_names:
                    processed_data = dataset[
                    dataset[self.config.seperable_groups]==nm
                    ][[self.config.covariate ,
                       self.config.random_effects_groups,
                       self.config.response]].copy(deep=True)
                    result = pd.concat(
                        [result,
                         self.run_lmem(processed_data, save_message='_' + nm)],
                        ignore_index=True
                    )
                result['Covariate'] = str(self.config.covariate)
                result['Response'] = self.config.response
                result = result[['Covariate','Response', 'Experiment', 'Intercept',
                                     'Intercept_pval', self.config.covariate,
                                     self.config.covariate+ '_pval', 'Summary']]
        return result

    def adjusted_evaluate(self, dataset=None):
        processed_data = dataset[[*self.config.covariate,
                                  self.config.random_effects_groups,
                                  self.config.response]].copy(deep=True)
        result = self.run_lmem(processed_data)
        if self.config.seperable_groups:
            group_names = list(set(dataset[
                                       self.config.seperable_groups
                                   ].to_list()))
            for nm in group_names:
                processed_data = dataset[
                    dataset[self.config.seperable_groups]==nm
                    ][self.config.covariate + [
                    self.config.random_effects_groups,
                    self.config.response]].copy(deep=True)
                result = pd.concat(
                    [result,
                     self.run_lmem(processed_data, save_message='_' + nm)],
                    ignore_index=True
                )
            result['Covariate'] = str(self.config.covariate)
            result['Response'] = self.config.response
            result = result[['Covariate','Response', 'Experiment', 'Intercept',
                             'Intercept_pval']+ self.config.covariate +
                            [c+ '_pval' for c in self.config.covariate]
                            + ['Summary']]
        return result

    def run_lmem(self, trial_data, save_message=None):
        md = smf.mixedlm(self.config.formula,
                         trial_data,
                         groups=trial_data[self.config.random_effects_groups],
                         re_formula=self.config.re_formula)
        try:
            res = md.fit(free=self.config.free, method=self.config.optimizers)
        except Exception as e:
            if self.config.adjusted:
                return pd.DataFrame(
                    dict({
                        'Experiment': save_message if save_message else 'AllData',
                        'Intercept': 'NonConvergent',
                        'Intercept_pval': 'NonConvergent',
                        'Summary': 'NonConvergent'
                    }, **{c:  'NonConvergent' for c in self.config.covariate},
                        **{c+ '_pval': 'NonConvergent' for c in
                           self.config.covariate}),
                    index=[0]
                )
            else:
                return pd.DataFrame(
                    {
                        'Experiment': save_message if save_message else 'AllData',
                        'Intercept': 'NonConvergent',
                        'Intercept_pval': 'NonConvergent',
                        self.config.covariate:  'NonConvergent',
                        self.config.covariate + '_pval': 'NonConvergent',
                        'Summary': 'NonConvergent'
                    },
                    index=[0]
                )
        if self.config.visualization:
            self.print_graphs(res, trial_data, save_message)

        if self.config.adjusted:
            return pd.DataFrame(
                dict({
                    'Experiment': save_message if save_message else 'AllData',
                    'Intercept': res.params['Intercept'],
                    'Intercept_pval': res.pvalues['Intercept'],
                    'Summary': res.summary().__str__()
                }, **{c:  res.params[c] for c in self.config.covariate},
                    **{c+ '_pval': res.pvalues[c] for c in
                       self.config.covariate}),
                index=[0]
            )
        else:
            return pd.DataFrame(
                {
                    'Experiment': save_message if save_message else 'AllData',
                    'Intercept': res.params['Intercept'],
                    'Intercept_pval': res.pvalues['Intercept'],
                    self.config.covariate:  res.params[self.config.covariate],
                    self.config.covariate + '_pval': res.pvalues[self.config.covariate],
                    'Summary': res.summary().__str__()
                },
                index=[0]
            )

    def print_graphs(self, mdf, data, message):
        name = "Pastel1"
        cmap = get_cmap(name)
        colors = cmap.colors
        plt.rc('axes', prop_cycle=(cycler('color', colors)))
        if type(self.config.vis_x)!=list:
            plt_x = [self.config.vis_x]
        else:
            plt_x = self.config.vis_x

        for plot_x in plt_x:
            for k in mdf.random_effects.keys():
                x = data[
                    data[self.config.random_effects_groups]==k
                    ][plot_x].to_list()
                x.sort()
                plt.plot(x,
                         mdf.random_effects[k]['Group'] +
                         mdf.params[plot_x] * np.asarray(x)
                         + mdf.random_effects[k][plot_x] +
                         mdf.params['Intercept'],
                         linewidth=2.5,
                         )

                plt.scatter(data[
                                data[self.config.random_effects_groups]==k
                                ][plot_x].to_list(),
                            data[data[self.config.random_effects_groups]==k
                                    ][self.config.response],
                         edgecolors='black')
            x = data[plot_x].to_list()
            x.sort()
            plt.plot(
                x,
                mdf.params[plot_x] *
                np.asarray(x) +
                mdf.params['Intercept'],
                color='black',
                linewidth=5
            )
            plt.xlabel(plot_x)
            plt.ylabel(self.config.response)
            plt.title('Longitudinal Anaylsis of ' + self.config.response + ' by ' + \
                      plot_x)
            if message:
                save_name = os.path.join(self.io_manager.root,
                                         self.config.response+'_'+plot_x+
                                         '_lmem_wRE'+message+'.pdf')
            else:
                save_name = os.path.join(self.io_manager.root,
                                         self.config.response+'_'+plot_x+
                                         '_lmem_wRE.pdf')
            plt.savefig(save_name)
            plt.savefig(save_name.replace('pdf','png'))
            plt.clf()

            for k in mdf.random_effects.keys():
                x = data[data[self.config.random_effects_groups]==k][
                    plot_x].to_list()
                plt.scatter(x,
                            data[data[self.config.random_effects_groups]==k][self.config.response],
                            edgecolors='black')
            x = data[plot_x].to_list()
            x.sort()
            plt.plot(
                x,
                mdf.params[plot_x] *
                np.asarray(x) +
                mdf.params['Intercept'],
                color='black',
                linewidth=5
            )
            plt.xlabel(plot_x)
            plt.ylabel(self.config.response)
            plt.title('Longitudinal Anaylsis of ' + self.config.response + ' by ' + \
                      plot_x)
            if message:
                save_name = os.path.join(self.io_manager.root,
                                     self.config.response+'_'+plot_x+
                                     '_lmem_woRE'+message+'.pdf')
            else:
                save_name = os.path.join(self.io_manager.root,
                                         self.config.response+'_'+plot_x+
                                         '_lmem_woRE.pdf')
            plt.savefig(save_name)
            plt.savefig(save_name.replace('pdf','png'))
            plt.clf()
