import React from 'react';
import Switch from 'react-router/Switch';
import withStyles from '@material-ui/core/styles/withStyles';
import Route from '../../../routes/RouteWrapper';
import Sidebar from '../common/Sidebar';
import AdInsertApp from '.';
import { styles } from '../common/layout';
import MessageBars from './MessageBars';

const AdRouter = ({ classes, match }) => (
  <div className={classes.root}>
    <Sidebar currentPath={match.path} />
    <div style={{ flexGrow: 1 }}>
      <Switch>
        {Route({ exact: true, path: '/', component: AdInsertApp })}
        {/* <Redirect to={'/404'}/> */}
      </Switch>
      <MessageBars />
    </div>
  </div>
);

export default withStyles(styles)(AdRouter);
