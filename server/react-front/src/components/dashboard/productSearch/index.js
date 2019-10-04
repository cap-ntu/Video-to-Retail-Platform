import React from 'react';
import * as PropTypes from 'prop-types';
import withStyles from '@material-ui/core/styles/withStyles';
import ContainerSearchPaper from './ContainerSearchPaper';
import ContainerInsertionPointPaper from './ContainerInsertionPointPaper';

const styles = {
  root: {
    width: '85%',
    margin: 'auto',
  },
};

const Index = ({ classes }) => (
  <div className={classes.root}>
    <ContainerSearchPaper />
    <ContainerInsertionPointPaper />
  </div>
);

Index.propTypes = {
  classes: PropTypes.object.isRequired,
};

export default withStyles(styles)(Index);
