import React from 'react';
import * as PropTypes from 'prop-types';
import InputBase from '@material-ui/core/InputBase';
import Paper from '@material-ui/core/Paper';
import IconButton from '@material-ui/core/IconButton';
import SearchIcon from '@material-ui/icons/SearchRounded';
import withStyles from '@material-ui/core/styles/withStyles';

const styles = theme => ({
  root: {
    padding: [[0.25 * theme.spacing.unit, 2 * theme.spacing.unit]],
    display: 'flex',
    alignItems: 'center',
    width: '100%',
    borderRadius: '0',
    boxShadow: theme.shadows[1],
    '&:hover': {
      boxShadow: theme.shadows[2],
    },
  },
  input: {
    marginLeft: 8,
    flex: 1,
  },
});

const TextSearch = ({ classes, textContent, onChange }) => (
  <Paper className={classes.root} elevation={1}>
    <InputBase
      className={classes.input}
      value={textContent}
      onChange={onChange}
      placeholder='Describe the Product'
    />
    <IconButton
      className={classes.iconButton}
      color='primary'
      aria-label='Search'
    >
      <SearchIcon />
    </IconButton>
  </Paper>
);

TextSearch.propTypes = {
  classes: PropTypes.shape({
    root: PropTypes.object.isRequired,
    input: PropTypes.object.isRequired,
  }).isRequired,
  textContent: PropTypes.string.isRequired,
  onChange: PropTypes.func.isRequired,
};

export default withStyles(styles)(TextSearch);
